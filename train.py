"""
Michael Patel
March 2021

Project description:
    To classify NBA team logos

File description:

"""
################################################################################
# Imports
from parameters import *
from model import build_cnn_mobilenet


################################################################################
# Main
if __name__ == "__main__":
    # ----- ETL ----- #
    # labels
    labels = []
    int2label = {}
    directories = os.listdir(TRAIN_DIR)
    for i in range(len(directories)):
        name = directories[i]
        labels.append(name)
        int2label[i] = name

    num_classes = len(labels)
    #print(f'Classes: {labels}')
    #print(f'Number of classes: {num_classes}')

    # create a text file with labels
    with open("labels.txt", "w") as f:
        for d in directories:
            f.write(d + "\n")

    # preprocessing
    # train generator
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=60,  # degrees
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.3,  # interval [-1.0, 1.0]
        height_shift_range=0.3,  # interval [-1.0, 1.0]
        brightness_range=[0.3, 1.0],  # 0 is no brightness, 1 is max brightness
        channel_shift_range=50.0,
        zoom_range=[0.7, 1.3],  # less than 1.0 is zoom in, more than 1.0 is zoom out
        rescale=1./255  # [0, 255] --> [0, 1]
    )

    train_generator = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE
    )

    # validation generator
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )

    validation_generator = validation_datagen.flow_from_directory(
        directory=VALIDATION_DIR,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE
    )

    # ----- MODEL ----- #
    model = build_cnn_mobilenet(num_classes=num_classes)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    model.summary()

    # ----- TRAIN ----- #
    history = model.fit(
        x=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(validation_generator))

    # plot accuracy
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(os.getcwd(), "training"))

    # save model
    model.save(SAVE_DIR)

    # ----- DEPLOY ----- #
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVE_DIR)
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
