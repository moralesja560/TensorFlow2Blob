import time
from datetime import datetime
x = 2561221.2151521

import time

start = time.time()
print("hello")
time.sleep(1)
end = time.time()
print(f"{(end - start):.3f} secs")
time.sleep(2)
end2 = time.time()
print(f"{(end2 - end):.3f} secs")
if (end2 - end) > 2:
	print("si es mayor")
