import random
import numpy as np
import math
from prettytable import PrettyTable
from numpy.linalg import solve
from scipy.stats import f, t

n = 15
m = 3

while True:
	x1_min = -20
	x1_max = 30
	x2_min = -20
	x2_max = 40
	x3_min = -20
	x3_max = 10
	x01 = (x1_max + x1_min) / 2
	x02 = (x2_max + x2_min) / 2
	x03 = (x3_max + x3_min) / 2

	dx1 = x1_max - x01
	dx2 = x2_max - x02
	dx3 = x3_max - x03

	xn = [[-1, -1, -1, 1, 1, 1, -1, 1, 1, 1],
		[-1, -1, 1, 1, -1, -1, 1, 1, 1, 1],
		[-1, 1, -1, -1, 1, -1, 1, 1, 1, 1],
		[-1, 1, 1, -1, -1, 1, -1, 1, 1, 1],
		[1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
		[1, -1, 1, -1, 1, -1, -1, 1, 1, 1],
		[1, 1, -1, 1, -1, -1, -1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		[-1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0, 0],
		[1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0, 0],
		[0, -1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0],
		[0, 1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0],
		[0, 0, -1.73, 0, 0, 0, 0, 0, 0, 2.9929],
		[0, 0, 1.73, 0, 0, 0, 0, 0, 0, 2.9929],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

	x1 = [x1_min, x1_min, x1_min, x1_min, x1_max, x1_max, x1_max, x1_max, - 1.73 * dx1 + x01, 1.73 * dx1 + x01, x01, x01, x01, x01, x01]
	x2 = [x2_min, x2_min, x2_max, x2_max, x2_min, x2_min, x2_max, x2_max, x02, x02, -1.73 * dx2 + x02, 1.73 * dx2 + x02, x02, x02, x02]
	x3 = [x3_min, x3_max, x3_min, x3_max, x3_min, x3_max, x3_min, x3_max, x03, x03, x03, x03, -1.73 * dx3 + x03, 1.73 * dx3 + x03, x03]

	x1x2 = [0] * 15
	x1x3 = [0] * 15
	x2x3 = [0] * 15
	x1x2x3 = [0] * 15
	x1kv = [0] * 15
	x2kv = [0] * 15
	x3kv = [0] * 15

	for i in range(15):
		x1x2[i] = round(x1[i] * x2[i], 3)
		x1x3[i] = round(x1[i] * x3[i], 3)
		x2x3[i] = round(x2[i] * x3[i], 3)
		x1x2x3[i] = round(x1[i] * x2[i] * x3[i], 3)
		x1kv[i] = round(x1[i] ** 2, 3)
		x2kv[i] = round(x2[i] ** 2, 3)
		x3kv[i] = round(x3[i] ** 2, 3)

	tmp_list_a = list(zip(x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3, x1kv, x2kv, x3kv))

	plan_table = PrettyTable()
	plan_table.field_names = ['X1', 'X2', 'X3', 'X1X2', 'X1X3', 'X2X3', 'X1X2X3', 'X1^2', 'X2^2', 'X3^2']
	print("Матриця планування з натуралізованими X:")
	for i in range(len(tmp_list_a)):
		plan_table.add_row(tmp_list_a[i])
	print(plan_table)

	def f123(X1, X2, X3):
		return 6.1 + 5.4*X1 + 0.2*X2 + 7.4*X3 + 8.8*X1*X1 + 0.8*X2*X2 + 5.0*X3*X3 + 4.5*X1*X2 + 0.5*X1*X3 + 4.7*X2*X3 + 2.6*X1*X2*X3 + random.randint(0, 10) - 5
	
	y = [[f123(tmp_list_a[j][0], tmp_list_a[j][1], tmp_list_a[j][2]) for _ in range(m)] for j in range(15)]


	plan_y = PrettyTable()
	plan_y.field_names = ['y1', 'y2', 'y3']
	print("Матриця планування значень Y:")
	for i in range(len(y)):
		plan_y.add_row(y[i])
	print(plan_y)

	aver_y = []
	for i in range(len(y)):
		aver_y.append(np.mean(y[i], axis=0))
	print("Середні значення Y:\n{}".format(np.array(list(map(lambda x:round(x, 5), aver_y)))))

	disp = []
	for i in range(len(y)):
		a = 0
		for k in y[i]:
			a += (k - np.mean(y[i], axis=0)) ** 2
		disp.append(a / len(y[i]))
	print("Дисперсія:\n{}".format(np.array(list(map(lambda x:round(x, 5), disp)))))


	def finds_value(num):
		a = 0
		for j in range(15):
			a += aver_y[j] * tmp_list_a[j][num - 1] / 15
		return a

	def a(f, s):
		a = 0
		for j in range(15):
			a += tmp_list_a[j][f - 1] * tmp_list_a[j][s - 1] / 15
		return a


	def finds_value(num):
		a = 0
		for j in range(15):
			a += aver_y[j] * tmp_list_a[j][num - 1] / 15
		return a
		
	def a(f, s):
		a = 0
		for j in range(15):
			a += tmp_list_a[j][f - 1] * tmp_list_a[j][s - 1] / 15
		return a

	my = sum(aver_y) / 15
	mx = []
	for i in range(10):
		number_lst = []
		for j in range(15):
			number_lst.append(tmp_list_a[j][i])
		mx.append(sum(number_lst) / len(number_lst))

	determinant1 = [[1, mx[0], mx[1], mx[2], mx[3], mx[4], mx[5], mx[6], mx[7], mx[8], mx[9]],
					[mx[0], a(1, 1), a(1, 2), a(1, 3), a(1, 4), a(1, 5), a(1, 6), a(1, 7), a(1, 8), a(1, 9), a(1, 10)],
					[mx[1], a(2, 1), a(2, 2), a(2, 3), a(2, 4), a(2, 5), a(2, 6), a(2, 7), a(2, 8), a(2, 9), a(2, 10)],
					[mx[2], a(3, 1), a(3, 2), a(3, 3), a(3, 4), a(3, 5), a(3, 6), a(3, 7), a(3, 8), a(3, 9), a(3, 10)],
					[mx[3], a(4, 1), a(4, 2), a(4, 3), a(4, 4), a(4, 5), a(4, 6), a(4, 7), a(4, 8), a(4, 9), a(4, 10)],
					[mx[4], a(5, 1), a(5, 2), a(5, 3), a(5, 4), a(5, 5), a(5, 6), a(5, 7), a(5, 8), a(5, 9), a(5, 10)],
					[mx[5], a(6, 1), a(6, 2), a(6, 3), a(6, 4), a(6, 5), a(6, 6), a(6, 7), a(6, 8), a(6, 9), a(6, 10)],
					[mx[6], a(7, 1), a(7, 2), a(7, 3), a(7, 4), a(7, 5), a(7, 6), a(7, 7), a(7, 8), a(7, 9), a(7, 10)],
					[mx[7], a(8, 1), a(8, 2), a(8, 3), a(8, 4), a(8, 5), a(8, 6), a(8, 7), a(8, 8), a(8, 9), a(8, 10)],
					[mx[8], a(9, 1), a(9, 2), a(9, 3), a(9, 4), a(9, 5), a(9, 6), a(9, 7), a(9, 8), a(9, 9), a(9, 10)],
					[mx[9], a(10, 1), a(10, 2), a(10, 3), a(10, 4), a(10, 5), a(10, 6), a(10, 7), a(10, 8), a(10, 9), a(10, 10)]]

	determinant2 = [my, finds_value(1), finds_value(2), finds_value(3), finds_value(4), finds_value(5), finds_value(6), finds_value(7), finds_value(8), finds_value(9), finds_value(10)]

	beta = list(map(lambda x:round(x, 5), solve(determinant1, determinant2)))
	print("\nРівняння регресії:")
	print("y = {} + {} * X1 + {} * X2 + {} * X3 + {} * Х1X2 + \n  + {} * Х1X3 + {} * Х2X3 + {} * Х1Х2X3 + {} * X11^2 + {} * X22^2 + {} * X33^2".format(beta[0], beta[1], beta[2], beta[3], beta[4], beta[5], beta[6], beta[7], beta[8], beta[9], beta[10]))
	y_i = [0] * 15
	
	for k in range(15):
		y_i[k] = beta[0] + beta[1] * tmp_list_a[k][0] + beta[2] * tmp_list_a[k][1] + beta[3] * tmp_list_a[k][2] + \
				beta[4] * tmp_list_a[k][3] + beta[5] * tmp_list_a[k][4] + beta[6] * tmp_list_a[k][5] + beta[7] * \
				tmp_list_a[k][6] + beta[8] * tmp_list_a[k][7] + beta[9] * tmp_list_a[k][8] + beta[10] * tmp_list_a[k][9]
	print("Експерементальні значення:\n{}".format(np.array(list(map(lambda x:round(x, 5), y_i)))))
	
	gp = max(disp) / sum(disp)
	gt = 0.3346
	print("\nКритерій Кохрена\nGp = {}".format(gp))
	if gp < gt:
		print("Дисперсія однорідна")
	else:
		print("Дисперсія неоднорідна")
		m += 1
		continue

	sb = sum(disp) / len(disp)
	sbs = (sb / (15 * m)) ** 0.5

	f3 = (m - 1) * n
	sign_coef = []
	insign_coef = []
	d = 11
	res = [0] * 11

	for j in range(11):
		t_pract = 0
		for i in range(15):
			if j == 0:
				t_pract += aver_y[i] / 15
			else:
				t_pract += aver_y[i] * xn[i][j - 1]
			res[j] = beta[j]
		if math.fabs(t_pract / sbs) < t.ppf(q=0.975, df=f3):
			insign_coef.append(beta[j])
			res[j] = 0
			d-=1
		else:
			sign_coef.append(beta[j])
		
	print("\nКритерій Стьюдента:")
	print("Значимі коефіцієнти регресії:", [round(i, 3) for i in sign_coef])
	print("Незначимі коефіцієнти регресії:", [round(i, 3) for i in insign_coef])
	y_st = []
	for i in range(15):
		y_st.append(res[0] + res[1] * x1[i] + res[2] * x2[i] + res[3] * x3[i] + res[4] * x1x2[i] + res[5] *
					x1x3[i] + res[6] * x2x3[i] + res[7] * x1x2x3[i] + res[8] * x1kv[i] + res[9] *
					x2kv[i] + res[10] * x3kv[i])
	print("Значення функції відгуку зі значущими коефіцієнтами:\n{}".format(np.array(list(map(lambda x:round(x, 5), y_st)))))
	
	print("\nКритерій Фішера")
	sad = m * sum([(y_st[i] - aver_y[i]) ** 2 for i in range(15)]) / (n - d)
	fp = sad / sb
	f4 = n - d
	print("Fp =", fp)
	if fp < f.ppf(q=0.95, dfn=f4, dfd=f3):
		print("Математична модель адекватна")
		break
	else:
		print("Математична модель неадекватна")
