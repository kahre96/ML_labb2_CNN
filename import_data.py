import tensorflow as tf

def import_data(img_h,img_w,batch_size,ds_path):
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        ds_path,
        validation_split=0.3,
        label_mode="categorical",
        subset="training",
        seed=1,
        image_size=(img_h, img_w),
        batch_size=batch_size,

    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        ds_path,
        validation_split=0.3,
        label_mode="categorical",
        subset="validation",
        seed=1,
        image_size=(img_h, img_w),
        batch_size=batch_size,

    )
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds
