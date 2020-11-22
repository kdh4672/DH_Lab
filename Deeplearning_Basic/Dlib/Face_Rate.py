class face_rate():
	def __init__(self,fa,fr,total):
		self.fa = fa
		self.fr = fr
		self.total = total

	def FAR(self):
		far = round(self.fa/self.total,3)
		return far

	def FRR(self):
		frr = round(self.fr/self.total,3)
		return frr
# x = face_rate(3,2,30)
# y = x.FAR()
# z = x.FRR()
# print(y,z)
