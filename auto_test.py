import autoencoder_phythia
import json


i = 0
sigma = 0
accu = []
scorez = []
while sigma < 20:
    res = autoencoder_phythia.autoencoder_pythia(sigma_1=0 ,sigma_2=sigma)
    sigma += 0.2
    scorez.append(res[0])
    accu.append(res[1])

f = open("sigma2_score.txt", "a")
score_string = " ".join(str(e) for e in scorez)
f.write(score_string)
f.close()


g = open("sigma2_acc.txt", "a")
acc_string = " ".join(str(e) for e in accu)
g.write(acc_string)
g.close()



