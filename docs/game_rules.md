# Azul 游戏规则

## 游戏概述
Azul是一个抽象的策略性瓷砖放置游戏，2-4名玩家通过收集和放置瓷砖来获得分数。

## 游戏组件
- 100个瓷砖（每种颜色20个）
- 5种颜色：蓝、黄、红、黑、白
- 每个玩家一个玩家板：
     - 准备区（5行三角形区域，从上到下1/2/3/4/5格）
     - 结算区（5x5格，固定颜色模式）
     - 扣分区（7格，从左到右分值：-1,-1,-2,-2,-2,-3,-3）
- 1个先手标记

## 游戏流程

### 1. 准备阶段
- 先手棋子放入待定区（第一轮随机选择起始玩家）
- 检查棋子池：
    - 如果新回合开始时棋子池为空，将废棋堆的棋子放回棋子池并洗混
- 将瓷砖随机分配到5个圆盘：
    - 每个圆盘尝试放4个棋子
    - 如果棋子池的棋子不够，有多少放多少
- 确定起始玩家：
    - 持有先手标记的玩家作为下一回合的起始玩家
    - 第一轮由随机选择的玩家开始

### 2. 回合阶段
- 玩家轮流行动，每次只能执行以下操作之一：
    a. 从一个圆盘选择一种颜色的所有棋子
    b. 从待定区选择一种颜色的所有棋子
- 选择后的处理：
    - 从圆盘选择时，其他颜色的棋子移入待定区
    - 从待定区选择时，如果待定区有先手棋子，该棋子会和选中的棋子一起被拿走
- 放置规则：
    - 玩家必须选择以下放置方式之一：
        a. 将所有选中的棋子放入准备区的某一行（从右到左严格填充）
        b. 将所有选中的棋子放入扣分区（从左到右填充）
    - 准备区限制：
        - 同一行只能放同色棋子
        - 如果某行已经在结算区有相同颜色，则不能在该行放置该颜色
        - 如果选择的棋子数量超过准备区该行剩余空格，多余的棋子必须放入扣分区
    - 先手棋子规则：
        - 获得先手棋子的玩家必须将其放入扣分区
        - 持有先手棋子的玩家将在下一回合首先行动

### 3. 结算阶段
当所有瓷砖都被选完时：
   - 准备区从上到下检查每一行：
     - 如果某行填满：
       - 将一颗棋子移到结算区对应行的对应颜色位置
       - 该行剩余的棋子移入废棋堆
     - 如果某行未填满，棋子保持原位
   - 计分：
     - 每当一颗棋子进入结算区时：
       - 检查是否能和水平或垂直方向形成连线（长度≥2）：
         - 如果能形成连线，每条线得分等于线的长度
         - 如果不能形成任何连线，这颗棋子得1分
     - 扣分区的棋子按格子显示的分值计算（负分），然后移入废棋堆
   - 结算区的棋子保留到下一回合
   - 未填满的准备区棋子保留到下一回合

### 4. 游戏结束
当任意玩家结算区完成一整行时，游戏结束。
最终得分 = 准备区在结算阶段的所有得分（包括扣分区的得分）以及结算区的奖励得分。
结算区的奖励得分只在游戏结束时候计算：
    ## 计分规则
    1. 完整行奖励：2分
    2. 完整列奖励：7分
    3. 颜色完成奖励：10分
