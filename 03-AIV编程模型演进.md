# AIV 编程模型演进：老架构 → MSD → RegBase

## 1. RegBase 是什么

950 Vector 核的**寄存器级 SIMD 编程模型**（Register-Based SIMD）。白皮书配套能力：UB 256KB + Register File，Vector 支持双发射。

**三段式结构**（`weight_quant_batch_matmul_v2_reg_base_common.h`，namespace `Reg = AscendC::Reg`）：

```cpp
__VEC_SCOPE__ {
    // 1. 加载：Reg::DataCopy 从 UB 到寄存器，顺手做解包/广播
    Reg::DataCopy<xType, LoadDist::DIST_UNPACK4_B8>(...)   // int4 加载时直接解包
    Reg::DataCopy<sType, LoadDist::DIST_BLK>(...)          // scale 按块广播
    // 2. 计算：RegTensor 上寄存器级运算
    Reg::Cast<half, xType, CAST_S4_TO_F16_TRAIT>(...)      // FP4 需 FP4→BF16→FP16 两段
    Reg::Add(vreg, vreg, offset, mask)
    Reg::Mul(vreg, vreg, scale, mask)
    // 3. 存回：StoreDist 写回 UB
    Reg::DataCopy<half, StoreDist::DIST_NORM_B16>(ub, vreg)
}
```

关键特征：`__VEC_SCOPE__` 作用域、`AddrReg`/`CreateAddrReg` 手工地址运算、`__local_mem__` 裸指针、`MaskReg`（MaskPattern::ALL）、三层手写循环每次处理 128 元素（`VECTOR_REG_WIDTH>>2`）、`bubpingpong_` 乒乓 buffer。

## 2. 三代 AIV 写法对比

| 维度 | 老架构（UB-to-UB 张量级） | MSD 模板（张量级 + 融合流） | RegBase（寄存器级） |
|---|---|---|---|
| 数据载体 | LocalTensor（UB） | LocalTensor（UB） | RegTensor（寄存器） |
| API 粒度 | Cast/Muls/Add 整张量 | 同上 + RowMax/RowSum/DivMC 等 | Reg::Cast/Add/Mul 单条向量指令 |
| 同步 | PipeBarrier + TQue/TBuf 事件 | 同左 + WaitForCube 流水 | 编译器调度 + 手工乒乓 |
| 地址管理 | 框架托管 | 框架托管 | AddrReg 手工运算 |
| int4 解包 | 单独 Cast 算子 | 单独 Cast 算子 | **加载顺手解包**（DIST_UNPACK4_B8） |
| 循环结构 | 单个大 API 调用 | 分块 + 多核流水 | 三层手写循环 ×128 元素 |

## 3. RegBase 的收益与代价

- **收益**：① UB 访问次数降一个量级（中间结果留在寄存器，不用每步读写 UB）；② 指令级调度空间——手工控制发射顺序，配合双发射填满两个槽位；③ 加载与解包/广播融合，省独立 pass
- **代价**：编程复杂度陡增——手工地址、手工 mask、手工乒乓、手工循环展开，正是"高级 API 重写为裸 API 获 10-15% 提升"这类简历故事的底层逻辑

## 4. 双发射（Dual-Issue）是什么

**定义**：发射宽度特性——Vector 核每周期可发射 **2 条指令**进入两套独立的执行单元。

- **前提**：ILP（指令级并行）——两条指令无数据依赖、无资源冲突
- **调度方式**：昇腾为编译器静态调度（类 VLIW），CCE 编译器负责配对；能否配成对，取决于代码里有没有足够多的独立依赖链
- **工程含义**：乒乓 buffer、手工展开循环、交错独立计算，本质都是**人为制造独立依赖链来喂满第二个发射槽**。这也是 RegBase 手写三层循环 ×128 元素的设计动机之一：循环体间无依赖，编译器才能双发射
- 与超标量的区别：不靠硬件动态检测（scoreboard/OOO），靠编译期排布，所以代码写法直接决定发射效率——写不出独立指令流，双发射就是空转

## 5. 面试口径

> "RegBase 是 950 Vector 的寄存器级编程模型，相比老架构的 UB-to-UB 张量级 API，它把中间结果留在寄存器、加载顺手解包 int4、手工管理地址和乒乓，核心收益是 UB 访问降量级 + 指令级调度空间。配合双发射特性——每周期发两条独立指令——手写循环制造独立依赖链喂满两个发射槽，这就是裸 API 重写能再榨 10-15% 的原因。代价是编程复杂度，所以模板库里它和 MSD 这类高层融合模板是互补关系：MSD 管算法结构，RegBase 管微架构榨干。"
