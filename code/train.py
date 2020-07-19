from config import *
from game import *
from datasets import *
import gc


def handle_batch():
    sketch_batch = []
    paint_batch = []
    for _ in range(batch_size):
        sketch_mat, paint_mat = handle_next()
        sketch_batch.append(sketch_mat)
        paint_batch.append(paint_mat)
    sketch_batch = np.stack(sketch_batch, axis=0)
    paint_batch = np.stack(paint_batch, axis=0)
    return sketch_batch, paint_batch


load_all_weights()

for index in range(50000):
    start = time.time()
    sketch_batch, paint_batch = handle_batch()
    train_on_batch(sketch_batch, paint_batch)
    if index % 200 == (200 - 1):
        save_all_weights()
    print(str(index) + '/50000 Currently ' + str(time.time() - start) + ' second for each iteration.')
    gc.collect()
