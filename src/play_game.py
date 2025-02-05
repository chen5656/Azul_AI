from game.environment import AzulEnv
from game.piece import PieceColor

def print_board_state(env):
    """打印当前游戏状态"""
    print("\n" + "="*50)
    print(f"当前玩家: {env.current_player}")
    print(f"回合数: {env.round_count}")
    print(f"玩家1得分: {env.player1_board.score}")
    print(f"玩家2得分: {env.player2_board.score}")
    
    # 打印圆盘状态
    print("\n圆盘状态:")
    for i, disk in enumerate(env.disks):
        pieces = [f"{p.color.name}" for p in disk]
        print(f"圆盘 {i}: {pieces}")
    
    # 打印待定区
    print("\n待定区:")
    pieces = [f"{p.color.name}" for p in env.waiting_area if not p.is_first]
    first_marker = any(p.is_first for p in env.waiting_area)
    print(f"棋子: {pieces}")
    if first_marker:
        print("先手标记在待定区")
    
    # 打印当前玩家的准备区和结算区
    current_board = env.player1_board if env.current_player == 1 else env.player2_board
    print("\n准备区:")
    for i, row in enumerate(current_board.prep_area):
        pieces = [p.color.name if p else "空" for p in row]
        print(f"行 {i}: {pieces}")
    
    print("\n结算区:")
    for i, row in enumerate(current_board.scoring_area):
        pieces = [p.color.name if p else "空" for p in row]
        print(f"行 {i}: {pieces}")
    
    print("\n扣分区:")
    pieces = [p.color.name if p else "空" for p in current_board.penalty_area]
    print(pieces)
    print("="*50 + "\n")

def get_valid_action():
    """获取用户输入的动作"""
    while True:
        try:
            print("\n请输入你的动作:")
            print("格式: source_type source_idx color target_type target_idx")
            print("说明:")
            print("source_type: 0=圆盘, 1=待定区")
            print("source_idx: 圆盘/待定区的索引")
            print("color: 1=蓝, 2=黄, 3=红, 4=黑, 5=白")
            print("target_type: 0=准备区, 1=扣分区")
            print("target_idx: 目标行/列的索引")
            print("示例: '0 1 3 0 3' 表示从圆盘1取红色棋子放到准备区第3行")
            
            action = input("请输入 (输入q退出): ")
            if action.lower() == 'q':
                return None
                
            source_type, source_idx, color, target_type, target_idx = map(int, action.split())
            
            # 转换颜色ID为名称
            color_names = {1: "蓝色", 2: "黄色", 3: "红色", 4: "黑色", 5: "白色"}
            source_names = {0: "圆盘", 1: "待定区"}
            target_names = {0: "准备区", 1: "扣分区"}
            
            # 显示动作描述
            print(f"\n准备执行: 从{source_names[source_type]}{source_idx}取{color_names[color]}棋子放到{target_names[target_type]}", end="")
            if target_type == 0:
                print(f"第{target_idx}行")
            else:
                print("")
                
            return (source_type, source_idx, color, target_type, target_idx)
        except (ValueError, IndexError, KeyError):
            print("输入格式错误，请重试")

def main():
    # 创建游戏环境
    env = AzulEnv()
    obs = env.reset()
    done = False
    
    print("欢迎来到Azul游戏!")
    print("游戏开始...")
    
    while not done:
        # 显示游戏状态
        print_board_state(env)
        
        # 获取当前玩家的动作
        action = get_valid_action()
        if action is None:
            break
            
        # 执行动作
        obs, reward, done, info = env.step(action)
        
        # 显示动作结果
        print(f"\n动作执行完成，奖励值: {reward}")
        if 'error' in info:
            print(f"错误: {info['error']}")
            
    # 游戏结束
    print("\n游戏结束!")
    print(f"玩家1最终得分: {env.player1_board.score}")
    print(f"玩家2最终得分: {env.player2_board.score}")
    winner = 1 if env.player1_board.score > env.player2_board.score else 2
    print(f"获胜者: 玩家{winner}")
    
if __name__ == "__main__":
    main() 