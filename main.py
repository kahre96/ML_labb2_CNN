from import_data import import_data
from buildmodel import build_model, generate_callback
import numpy as np

if __name__ == '__main__':

    img_h = 128
    img_w = 128
    batch_size = 64
    epochs = 200
    patience = 25

    # create a training and validation dataset
    train_ds, val_ds = import_data(128, 128, 32, "images")

    model = build_model(img_h, img_w)

    model.summary()

    callbacks = generate_callback(patience)

    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data= val_ds,
        batch_size=batch_size

    )

    # prints results of best epoch
    best_epochs = np.argmin(history.history['val_loss'])
    print("val_accuracy", history.history['val_accuracy'][best_epochs])
    print("val_loss", history.history['val_loss'][best_epochs])
    print("accuracy", history.history['accuracy'][best_epochs])
    print("loss", history.history['loss'][best_epochs])