import os
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1.指明各类图片文件路径（相对路径，注意：请把py文件放在NEU-DET文件中并保持与下列图片文件平级）
# 注释：训练集的各类图片包括六种（crazing、inclusion、patches、 pitted_surface、rolled-in_scale、scratches）
train_crazing_dir = os.path.join('../NEU-DET/train/images/crazing')  # "钢筋细裂纹"
train_inclusion_dir = os.path.join('../NEU-DET/train/images/inclusion/')  # "钢筋夹杂物"
train_patches_dir = os.path.join('../NEU-DET/train/images/patches/')  # "钢筋锈蚀"
train_pitted_surface_dir = os.path.join('../NEU-DET/train/images/pitted_surface/')  # "钢筋麻面"
train_rolledin_scale_dir = os.path.join('../NEU-DET/train/images/rolled-in_scale/')  # "钢筋压入氧化铁皮"
train_scratches_dir = os.path.join('../NEU-DET/train/images/scratches/')  # "钢筋划痕"
# 注释：返回指定的文件夹包含的文件或文件夹的名字的列表
train_crazing_names = os.listdir(train_crazing_dir)
train_inclusion_names = os.listdir(train_inclusion_dir)
train_patches_names = os.listdir(train_patches_dir)
train_pitted_surface_names = os.listdir(train_pitted_surface_dir)
train_rolledin_scale_names = os.listdir(train_rolledin_scale_dir)
train_scratches_names = os.listdir(train_scratches_dir)
# 注释：显示各类图片的数量都是240张
print('total training crazing images:', len(train_crazing_names))  # len函数是求图片数量的
print('total training inclusion images:', len(train_inclusion_names))
print('total training patches images:', len(train_patches_names))
print('total training pitted_surface images:', len(train_pitted_surface_names))
print('total training rolled_in_scale images:', len(train_rolledin_scale_names))
print('total training scratches images:', len(train_scratches_names))
# 2.输入模型信息
model = tf.keras.models.Sequential([
    # 卷积1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    # 池化2
    tf.keras.layers.MaxPooling2D(2, 2),
    # 卷积2
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # 池化2
    tf.keras.layers.MaxPooling2D(2, 2),
    # 卷积3
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # 池化3
    tf.keras.layers.MaxPooling2D(2, 2),
    #扁平化
    tf.keras.layers.Flatten(),
    #全连接
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax'),

])

# 3.输出模型信息
model.summary()

# 注释：model.compile()告知训练时用的优化器、损失函数和准确率评测标准optimizer =优化器， loss =损失函数， metrics = ["准确率”]
# 注释：使用多分类交叉熵作为损失函数
#4.编译模型
model.compile(loss='categorical_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])

# 注释：ImageDataGenerator()是keras.preprocessing.image模块中的图片生成器（可以每一次给模型“喂”一个batch_size大小的样本数据，
# 注释：同时也可以在每一个批次中对这batch_size个样本数据进行增强，扩充数据集大小，增强模型的泛化能力。比如进行旋转，变形，归一化等等）
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#5.规范训练集和测试集
#将图片做成一个大小
train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    directory = '../NEU-DET/train/images/',
    target_size = (224, 224),
    batch_size = 60,
    class_mode = 'categorical'
)
val_generator = train_datagen.flow_from_directory(
    directory = '../NEU-DET/validation/images',
    target_size = (224, 224),
    batch_size = 16,
    class_mode = 'categorical'
)



# 6.模型训练部分和输入验证集
history = model.fit(
    train_generator,
    steps_per_epoch=16,
    epochs=30,
    verbose=1,
    validation_data = val_generator,
    shuffle=True
)

#训练好的模型的准确率
score = model.evaluate(val_generator, verbose=2)
# 输出测试集的准确率信息
print('-----------------------\nEvaluating the trained model.')
print('Test loss:', score[0])
print('Test accuracy:', score[1])









