class BMI_Calculator:
    def __init__(self, weight_kg, height_m):
        self.weight_kg = weight_kg
        self.height_m = height_m

    def calculate_bmi(self):
        return self.weight_kg / (self.height_m ** 2)

    def interpret_bmi(self):
        bmi = self.calculate_bmi()
        if bmi < 18.5:
            return "体重过轻"
        elif 18.5 <= bmi < 24.9:
            return "正常体重"
        elif 24.9 <= bmi < 29.9:
            return "超重"
        else:
            return "肥胖"

# 用户输入体重（千克）和身高（米）
weight = float(input("请输入您的体重（千克）："))
height = float(input("请输入您的身高（米）："))

# 创建BMI计算器对象
bmi_calculator = BMI_Calculator(weight, height)

# 计算BMI指数
bmi = bmi_calculator.calculate_bmi()

# 解释BMI指数
bmi_category = bmi_calculator.interpret_bmi()

# 打印结果
print(f"您的BMI指数为：{bmi:.2f}")
print(f"您的体重属于：{bmi_category}")
