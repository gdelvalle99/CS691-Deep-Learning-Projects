import tensorflow as tf
import tensorflow_addons as tfa
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cv2

def PatchLayer(filters,size,norm=True):
    initializer = tf.random_normal_initializer(0.,0.02)
    results = tf.keras.Sequential()
    results.add(tf.keras.layers.Conv2D(filters,size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
    if(norm == True):
        results.add(tfa.layers.InstanceNormalization(axis=-1))
    results.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    return results

def GenLayer(filters,size,stride=2,norm=True):
    initializer = tf.random_normal_initializer(0.,0.02)
    results = tf.keras.Sequential()
    results.add(tf.keras.layers.Conv2D(filters,size,strides=stride,padding='same',kernel_initializer=initializer,use_bias=False))
    if(norm == True):
        results.add(tfa.layers.InstanceNormalization(axis=-1))
    results.add(tf.keras.layers.ReLU())
    return results

def ResnetBlock(filters,size):
    initializer = tf.random_normal_initializer(0.,0.02)
    results = tf.keras.Sequential()
    results.add(tf.keras.layers.Conv2D(filters,size,padding='same',kernel_initializer=initializer))
    results.add(tfa.layers.InstanceNormalization(axis=-1))
    results.add(tf.keras.layers.ReLU())
    results.add(tf.keras.layers.Conv2D(filters,size,padding='same',kernel_initializer=initializer))
    results.add(tfa.layers.InstanceNormalization(axis=-1))
    return results

def Discriminator():
    initializer = tf.random_normal_initializer(0.,0.02)
    input = tf.keras.layers.Input(shape=[256,256,3], name='input_image')
    #target = tf.keras.layers.Input(shape=[256,256,3], name='target_image')

    #x = tf.keras.layers.concatenate([input])#,target])
    x = input
    patch1 = PatchLayer(64,4,False)(x)
    patch2 = PatchLayer(128,4)(patch1)
    patch3 = PatchLayer(256,4)(patch2)
    #zero_pad1 = tf.keras.layers.ZeroPadding2D()(patch3)
    patch4 = PatchLayer(512,4)(patch3)
    #zero_pad1 = tf.keras.layers.ZeroPadding2D()(patch3)
    #conv1 = tf.keras.layers.Conv2D(512,4,strides=2,kernel_initializer=initializer,use_bias=False)(patch3)
    #in1 = tfa.layers.InstanceNormalization(axis=-1)(conv1)
    #lrelu1 = tf.keras.layers.LeakyReLU()(in1)

    #zero_pad2 = tf.keras.layers.ZeroPadding2D()(patch4)
    conv2 = tf.keras.layers.Conv2D(512,4,strides=1,padding='same',kernel_initializer=initializer,use_bias=False)(patch4)
    in2 = tfa.layers.InstanceNormalization(axis=-1)(conv2)
    lrelu2 = tf.keras.layers.LeakyReLU()(in2)

    last = tf.keras.layers.Conv2D(1,4,strides=1,padding='same',kernel_initializer=initializer)(lrelu2)
    return tf.keras.Model(inputs=input,outputs=last)

def Generator():
    initializer = tf.random_normal_initializer(0.,0.02)
    input = tf.keras.layers.Input(shape=[256,256,3],name='input_image')

    conv1 = GenLayer(64,7,stride=1)(input)
    conv2 = GenLayer(128,3)(conv1)
    conv3 = GenLayer(256,3)(conv2)

    res1 = ResnetBlock(256,3)(conv3)
    con1 = tf.keras.layers.concatenate([res1,conv3])
    res2 = ResnetBlock(256,3)(con1)
    con2 = tf.keras.layers.concatenate([res2,con1])
    res3 = ResnetBlock(256,3)(con2)
    con3 = tf.keras.layers.concatenate([res3,con2])
    res4 = ResnetBlock(256,3)(con3)
    con4 = tf.keras.layers.concatenate([res4,con3])
    res5 = ResnetBlock(256,3)(con4)
    con5 = tf.keras.layers.concatenate([res5,con4])
    res6 = ResnetBlock(256,3)(con5)
    con6 = tf.keras.layers.concatenate([res6,con5])
    res7 = ResnetBlock(256,3)(con6)
    con7 = tf.keras.layers.concatenate([res7,con6])
    res8 = ResnetBlock(256,3)(con7)
    con8 = tf.keras.layers.concatenate([res8,con7])
    res9 = ResnetBlock(256,3)(con8)
    con9 = tf.keras.layers.concatenate([res9,con8])

    conv4 = tf.keras.layers.Conv2DTranspose(128,3,strides=2,padding='same',kernel_initializer=initializer)(con9)
    in1 = tfa.layers.InstanceNormalization(axis=-1)(conv4)
    relu1 = tf.keras.layers.ReLU()(in1)

    conv5 = tf.keras.layers.Conv2DTranspose(64,3,strides=2,padding='same',kernel_initializer=initializer)(relu1)
    in2 = tfa.layers.InstanceNormalization(axis=-1)(conv5)
    relu2 = tf.keras.layers.ReLU()(in2)

    conv6 = tf.keras.layers.Conv2D(3,7,padding='same',kernel_initializer=initializer)(relu2)
    in3 = tfa.layers.InstanceNormalization(axis=-1)(conv6)
    out = tf.keras.layers.Activation('tanh')(in3)

    return tf.keras.Model(inputs=input,outputs=out)

#mod = Discriminator()
#model_gen = Generator()
#model_gen.summary()
#tf.keras.utils.plot_model(model_gen, show_shapes=True, dpi=64)

genA = Generator()
genB = Generator()
discA = Discriminator()
discB = Discriminator()


LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse_obj = tf.keras.losses.MeanSquaredError()
#mae_obj = tf.keras.losses.MeanAbsoluteError()

def discriminator_loss(real,fake):
    real_loss = mse_obj(tf.ones_like(real),real)
    fake_loss = mse_obj(tf.zeros_like(fake),fake)
    total_loss = real_loss + fake_loss
    return total_loss *.5

def generator_loss(fake):
    return mse_obj(tf.ones_like(fake),fake)

def cyc_loss(real,cycle):
    loss1 = tf.reduce_mean(tf.abs(real-cycle))
    return LAMBDA * loss1

def id_loss(real,same):
    loss = tf.reduce_mean(tf.abs(real-same))
    return LAMBDA * 0.5 * loss


genA_opt = tf.keras.optimizers.Adam(2e-4,beta_1=0.5)
genB_opt = tf.keras.optimizers.Adam(2e-4,beta_1=0.5)

discA_opt = tf.keras.optimizers.Adam(2e-4,beta_1=0.5)
discB_opt = tf.keras.optimizers.Adam(2e-4,beta_1=0.5)

tf.keras.utils.plot_model(
    discA, to_file='discriminator_model.png', show_shapes=True)

tf.keras.utils.plot_model(
    genA, to_file='generator_model.png', show_shapes=True)
@tf.function
def train_step(real_x,real_y):
    print("here")
    with tf.GradientTape(persistent=True) as tape:
        fake_y = genA(real_x,training=True)
        cycled_x = genB(fake_y,training=True)

        fake_x = genB(real_y,training=True)
        cycled_y = genA(fake_x,training=True)

        same_x = genB(real_x,training=True)
        same_y = genA(real_y,training=True)

        disc_real_x = discA(real_x, training=True)
        disc_real_y = discB(real_y, training=True)

        disc_fake_x = discA(fake_x, training=True)
        disc_fake_y = discB(fake_y, training=True)

        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = cyc_loss(real_x,cycled_x) + cyc_loss(real_y,cycled_y)

        total_gen_g_loss = gen_g_loss + total_cycle_loss + id_loss(real_y,same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + id_loss(real_x,same_x)

        disc_x_loss = discriminator_loss(disc_real_x,disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
        print("Generator A Loss:", total_gen_g_loss)
        print("Discriminator A Loss:", disc_x_loss)

    #print("after losses")
    gen_g_gradients = tape.gradient(total_gen_g_loss, genA.trainable_variables)
    gen_f_gradients = tape.gradient(total_gen_f_loss, genB.trainable_variables)

    disc_x_gradients = tape.gradient(disc_x_loss, discA.trainable_variables)
    disc_y_gradients = tape.gradient(disc_y_loss, discB.trainable_variables)

    #print("after gradients")
    genA_opt.apply_gradients(zip(gen_g_gradients,genA.trainable_variables))
    genB_opt.apply_gradients(zip(gen_f_gradients,genB.trainable_variables))
    discA_opt.apply_gradients(zip(disc_x_gradients,discA.trainable_variables))
    discB_opt.apply_gradients(zip(disc_y_gradients,discB.trainable_variables))
    #print("after applying")


EPOCHS = 2000
BUFFER_SIZE = 1000
BATCH_SIZE = 1
nt = 0
def generate_images(model, test_input):
  global nt
  prediction = model(test_input)

  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.savefig(str(nt) + '.png')
  plt.close()
  nt += 1

# normalizing the images to [-1, 1]
def resize(input_image,  height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image

def normalize(input_image):
  input_image = (input_image / 127.5) - 1

  return input_image

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)


  input_image = tf.cast(image, tf.float32)

  return input_image



@tf.function()
def random_jitter(input_image):
  # resizing to 286 x 286 x 3
  input_image = resize(input_image, 256, 256)

  # randomly cropping to 256 x 256 x 3

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)


  return input_image

def load_image_train(image_file):
  input_image = load(image_file)
  input_image = random_jitter(input_image)
  input_image = normalize(input_image)

  return input_image

path = 'vangogh2photo/'
trainx_dataset = tf.data.Dataset.list_files(path+'trainB/*.jpg')
trainy_dataset = tf.data.Dataset.list_files(path+'trainA/*.jpg')
trainx_dataset = trainx_dataset.map(load_image_train)
trainx_dataset = trainx_dataset.shuffle(BUFFER_SIZE)
trainx_dataset = trainx_dataset.batch(BATCH_SIZE)
trainy_dataset = trainy_dataset.map(load_image_train)
trainy_dataset = trainy_dataset.shuffle(BUFFER_SIZE)
trainy_dataset = trainy_dataset.batch(BATCH_SIZE)

sample = next(iter(trainx_dataset))
sampley = next(iter(trainy_dataset))

IMG_HEIGHT = 256
IMG_WIDTH = 256

checkpoint_path = "./checkpoints/van"

ckpt = tf.train.Checkpoint(genA=genA,
                           genB=genB,
                           discA=discA,
                           discB=discB,
                           genA_opt=genA_opt,
                           genB_opt=genB_opt,
                           discA_opt =discA_opt,
                           discB_opt=discB_opt)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')
#image_a_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#train_a_gen = image_a_gen.flow_from_directory(directory='monet2photo/trainA',shuffle=True,target_size=(IMG_HEIGHT,IMG_WIDTH))
#image_b_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#train_b_gen = image_b_gen.flow_from_directory(directory='monet2photo/trainB',shuffle=True,target_size=(IMG_HEIGHT,IMG_WIDTH))
#sample = next(train_a_gen)
#generate_images(genA,sample)
#t = 0
ds = tf.data.Dataset.zip((trainx_dataset,trainy_dataset))
for epoch in range(EPOCHS):
    print(epoch)
    #start = time.time()
    n = 0
    t = 0
    image_x,image_y = next(iter(ds))
    #image_x, image_y = load_image_train()
    #for image_x, image_y in ds.as_numpy_iterator():
        #if t == 10:
        #    break
        #print(image_x.shape,image_y.shape)
    train_step(image_x,image_y)
        #print(t)
        #t+=1
        #print(n)
    if n % 10 == 0:
        print('.',end='')
    n+=1
    t+=1

    #print(n)

    ##clear_output(wait=True)

    if (epoch + 1) % 10 == 0:
        generate_images(genA,sample)
        #generate_images(genB,sample)

    if (epoch + 1) % 5 == 0:
        #generate_images(genA,sample)
        #generate_images(genB,sampley)
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
        #print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
