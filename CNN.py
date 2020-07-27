import tensorflow as tf

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(32,32,3)),
        tf.keras.layers.MaxPool2D(2,2),
        
        tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
        tf.keras.layers.MaxPool2D(2,2),
        
        tf.keras.layers.Conv2D(128,(3,3),activation="relu"),
        tf.keras.layers.MaxPool2D(2,2),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256,activation="relu"),
        tf.keras.layers.Dense(4,activation="softmax")
])

    
def print_model_summary():
    print("model summary 👀")
    model.summary()
    
class Callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs['acc']>0.98):
      print("\nReached 98% accuracy so cancelling training 😁")
      self.model.stop_training = True
    
def train_model(x_train,y_train,epochs):
    print("start traing 🐱‍👤 ...")
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
    callback = Callback()
    model.fit(x_train,y_train,epochs=epochs,callbacks=[callback])
    print("finshed traing 🚀🚀 .")
    
def evaluate_model(x_test,y_test): 
    print("Evaluate on test data 🤔")
    results = model.evaluate(x_test,y_test,batch_size=1)
    print("test accuracy",round(results[1]*100),sep=" : ",end=" % \n")
    
    
    