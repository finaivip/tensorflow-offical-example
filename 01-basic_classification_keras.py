# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# 1.准备数据
# 下载或加载数据
# 如果已经下载了，会将数据缓存在:~/.keras/datasets/fashion-mnist/train-labels-idx1-ubyte.gz
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 存储标签对应的类别名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# 0	T 恤衫/上衣
# 1	裤子
# 2	套衫
# 3	裙子
# 4	外套
# 5	凉鞋
# 6	衬衫
# 7	运动鞋
# 8	包包
# 9	踝靴

# 看看数据样子
print('train_image shape:{}, train_label shape: {}, test_image shape:{}, test_label shape: {}'
      .format(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape))
print('train labeles: {}, test_labels: {}'.format(set(train_labels), set(test_labels)))

# 预处理数据
print(train_images[0])
# 画图看看数据
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.pause(15)
# plt.close()

# 将数据处理为0~1之间
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

#plt.pause(15)
plt.close()

# 2.构建并编译模型
# 构建
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),   # 28 * 28 = 784
    keras.layers.Dense(128, activation=tf.nn.relu),   # 全连接网络  784 * 128 + 128 = 100480
    keras.layers.Dense(10, activation=tf.nn.softmax)  # 10表示最终分类的10个类别  128 * 10 + 10 = 1290
])
model.summary()
# Flatten将二维数组打平成一维，值修改数据的格式，没有参数

# 编译
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 编译的含义是给模型添加一些设置项，才能进行训练，包括：
# 1）损失函数: 衡量模型在训练期间的准确率，尽可能缩小该函数，引导模型朝正确的方向优化
# 2) 优化器: 根据模型看到的数据和损失函数，更新模型的方式
# 3) 指标(metrics)：监控训练和测试的步骤，上面使用的是准确率，即被正确分类的比例

# 3. 训练和评估模型
model.fit(train_images, train_labels, epochs=5)

# 主要将测试数据送入到模型，并使模型拟合(fit)训练数据; 准确率88%

# 模型评估
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('loss:{}, accuracy:{}'.format(test_loss, test_acc))  # 87%
# 在测试数据上准确率低于训练数据，这种现象说明出现了过拟合。


# 4. 模型预测
predictions = model.predict(test_images)
print(predictions[0])
print('predic: {}, label: {}'.format(np.argmax(predictions[0]), test_labels[0]))