import random

class Character:
    """角色实体类"""

    def __init__(self, user_id: int = 0, name: str = "", role: str = ""):
        # 如果没有提供参数，创建一个空对象供from_dict使用
        if not user_id and not name and not role:
            user_id = "999999"
            name = "初始化默认对象"
            role = ""

        self.user_id = user_id
        self.name = name
        self.role = role

        # 初始化基础属性
        self.age = random.randint(18, 45)
        self.gender = random.choice(["男", "女", "其他"])
        self.hometown = random.choice([
            "北方雪国", "沙漠绿洲", "滨海城邦", "森林部落",
            "高原王国", "地下都市", "浮空岛屿", "沼泽村落"
        ])

        # 初始化基础属性（3D6规则）
        self.STR = self._roll_3d6()  # 力量
        self.CON = self._roll_3d6()  # 体质
        self.SIZ = self._roll_2d6() + 6  # 体型
        self.DEX = self._roll_3d6()  # 敏捷
        self.APP = self._roll_3d6()  # 外貌
        self.INT = self._roll_2d6() + 6  # 智力
        self.POW = self._roll_3d6()  # 意志
        self.EDU = self._roll_2d6() + 6  # 教育

        # 初始化派生属性
        self.MOV = 9 if self.STR < self.SIZ else 10
        self.HP = self.CON + self.SIZ  # 当前HP
        self.SAN_max = self.POW  # 最大理智
        self.SAN = self.SAN_max  # 当前理智
        self.MP = max(8, (self.POW // 3) + self.EDU // 4)
        self.LUCK = self._roll_3d6()

    @property
    def HP_max(self):
        """计算最大生命值"""
        return int(self.CON + self.SIZ)  # 实时计算

    @staticmethod
    def _roll_3d6():
        return sum(sorted([random.randint(1, 6) for _ in range(3)])[1:])  # 取中间两个骰子

    @staticmethod
    def _roll_2d6():
        return sum([random.randint(1, 6) for _ in range(2)])