import numpy as np
from matplotlib import pyplot as plt
scale = [5, 10, 15, 20, 25, 30, 50, 70, 90]
jpeg_ss = [0.741778419,	0.794939021,	0.834540755,	0.85469251,	0.865181755,	0.876962592,	0.904983243,	0.929308726,	0.960448597]
final_ss = [0.717517132,	0.78996548,	0.836614245,	0.856861471,	0.867836105,0.87969008,	0.907297052,	0.930152147,	0.955735188]




plt.subplot(121)
plt.scatter(scale, jpeg_ss)
plt.plot(scale, jpeg_ss)
plt.xlabel('Q')
plt.ylabel('ssim')
plt.title('jpeg')

plt.subplot(122)
plt.scatter(scale, final_ss)
plt.plot(scale, final_ss)
plt.xlabel('Q')
plt.ylabel('ssim')
plt.title('final')

plt.show()