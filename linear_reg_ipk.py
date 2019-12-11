'''
    Linear Regression
    Rumus umum : y = m*x + b

    Program by prokoding
'''
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

#data yang sudah kita punya
jam = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5],dtype=np.float64)
ipk = np.array([2, 2.5, 3, 3.3, 3.4, 3.5, 3.7, 3.9, 4],dtype=np.float64)

class SimpleReg:

    def __init__(self, x, y):
        self.jam = x
        self.ipk = y

    def start(self):
        m = ( ( (mean(self.jam) * mean(self.ipk)) - mean(self.jam * self.ipk)) /
                ((mean(self.jam)**2) - mean(self.jam**2) ) )
        
        b = mean(self.ipk) - m * mean(self.jam)

        return m, b

    def squared_error(self, ipk_linear):
        return sum((ipk_linear - self.ipk)**2)

    def check_corr(self, ipk_linear):
        ipk_mean = []
        for i in self.ipk:
            ipk_mean.append(mean(self.ipk))
        se_best_fit = self.squared_error(ipk_linear)
        se_mean_original = self.squared_error(ipk_mean)
        return 1-(se_best_fit / se_mean_original)

#menggunakan classifier Simple Regression yang sudah dibuat
clf = SimpleReg(jam, ipk)
#mendapatkan nilai m dan b
m, b = clf.start()
#membuat garis berdasarkan rumus y = mX + b
reg_line = []
for x in jam:
    reg_line.append(m*x + b)

#membuat prediksi berapa ipk nya jika belajar 0.5 jam perhari
prediksi_jam = 0.5
prediksi_ipk = m * prediksi_jam + b
print('Jika belajar selama ',prediksi_jam,' jam sehari maka ipk yang akan di dapat : ',round(prediksi_ipk,2))

#mendapatkan nilai keterhubungan
correlation = clf.check_corr(reg_line)
print('Keterhubungan antara lama belajar dengan ipk yang baik adalah : ',round(correlation,2)*100,'%')

#membuat visualisasi prediksi dengan titik berwarna biru
plt.scatter(prediksi_jam, prediksi_ipk,color='b')

#membuat visualisasi data yang tersedia
plt.scatter(jam,ipk,color = 'g')

#membuat garis linear atau best fit slope
plt.plot(jam, reg_line)
plt.xlabel('Lama belajar')
plt.ylabel('IPK')
plt.show()




    
