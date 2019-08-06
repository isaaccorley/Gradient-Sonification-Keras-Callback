# Gradient Sonification Keras Callback
 Keras Callback for Auralization of Gradient Norms

Inspired heavily by [Christian Perone's blog post and Pytorch implementation](http://blog.christianperone.com/2019/08/listening-to-the-neural-network-gradient-norms-during-training/)

### 1. Setup and compile your model
```python
model = Sequential([...])
model.compile(loss='categorical_crossentropy',
              optimizer=opt(lr=lr),
              metrics=['accuracy'])
              
```

### 2. Define the callback
```python
fs = 44100
duration = 0.01
freq = 200.0

grad_son = GradientSonification(path='sample',
                                model=model,
                                fs=fs,
                                duration=duration,
                                freq=freq,
                                plot=True)
                                
```

### 3. Recompile your model with the new metrics
```python
model.compile(loss='categorical_crossentropy',
              optimizer=opt(lr=lr),
              metrics=['accuracy'] + grad_son.metrics)
              
```

### 4. Have you model.fit today?
```python
model.fit(X_train, y_train,
          batch_size=32,
          epochs=5,
          callbacks=[grad_son])

```
