import tensorflow as tf

model = tf.keras.models.load_model("models/face_classifier.h5")

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "testimages",
        label_mode="categorical",
        seed=1,
        image_size=(128, 128),

    )

score = model.evaluate(test_ds)

print('Test loss:', score[0])
print('Test accuracy:', score[1])