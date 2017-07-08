import tensorflow as tf
import numpy as np

m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],
                  [2]])
product = tf.matmul(m1, m2)

sess = tf.Session()
ans = sess.run(product)

print(ans)
sess.close()
