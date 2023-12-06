from datetime import datetime


acc_hr = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
now = datetime.now()
hora_consumer = int(now.strftime("%H"))
item = 102
acc_hr[int(hora_consumer)] = item
acc_hr[int(hora_consumer)-1] = item*5
accumlated_consumer = sum(acc_hr.values())
print(f"el valor que puse es {item}, la suma que debe dar el diccionario es {item+(item*5)} y realmente es {accumlated_consumer}. El acceso a la variable es {acc_hr[int(10)]}")



