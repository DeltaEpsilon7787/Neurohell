# -*- coding: utf-8 -*-

import CNN

alpha = CNN.Recognizer.Recognizer('CNN.json', start_anew=True)
beta = CNN.Introspection.Introspector(alpha.model)

gamma = alpha.train_model(epochs=1024, verbose=1)