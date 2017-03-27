class Test:
	def __init__(self, a, b):
		self.a = a
		self.b = b
		if True:
			self.c = 4
	def func1(self):
		self.c = 10
	def func2(self):
		print(self.c)

test = Test(100, 5)
test.func2()