import itertools
import random
import string
from tqdm import tqdm


listString = []
for i in tqdm(range(200000000)):
    # Create a rando string that strarts with the index
    tempString = str(i) + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    listString.append(tempString)

# Write to file
with open("outputs/captionKey.txt", "w") as f:
    for string in listString:
        f.write(string + "\n")

# Use linecache to read random lines and measure the time
import linecache
import time

startTime = time.time()
for i in range(1000000):
    line = linecache.getline("outputs/captionKey.txt", random.randint(0, 20000000) + 1)
print(f"Time taken to read a line: {(time.time() - startTime) / 1000000} sec")
print(linecache.getline("outputs/captionKey.txt", 152 + 1))


startTime = time.time()
for i in range(1000000):
    index = random.randint(0, 20000000)
    # Skip to the desired line number
    with open("outputs/captionKey.txt") as f:
        lines = itertools.islice("outputs/captionKey.txt", index-1, index)
        line = next(lines, '')

print(f"Time taken to read a line: {(time.time() - startTime) / 1000000} sec")
with open("outputs/captionKey.txt") as f:
    lines = itertools.islice("outputs/captionKey.txt", 152-1, 152+1)
    print(len(lines))
    print(next(lines, ''))

