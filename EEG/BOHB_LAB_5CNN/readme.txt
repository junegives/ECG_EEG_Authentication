Bohb가 찾은 Best found configuration 보다 내가 찾은 configuration이 더 성능이 높게 나온 sub들

Sub 5 :

Bohb found configuration
    Best found configuration: {'dropout_rate': 0.5001856856580251, 'lr': 0.0006140608741342609, 'lstm_units': 42, 'num_conv_layers': 2, 'num_fc_layers': 2, 'num_fc_units_1': 92, 'num_filters_1': 61, 'optimizer': 'Adam', 'num_fc_units_2': 77, 'num_filters_2': 150}
    A total of 250 unique configurations where sampled.
    A total of 300 runs where executed.
    Total budget corresponds to 200.0 full function evaluations.

    val_loss: 0.15898266434669495
    test_loss: 0.2966841161251068
    test_acc: 0.8590047359466553
    precision: 0.9423076923076923
    recall: 0.46445497630331756
    specificity 0.990521327014218
    sensitivity :  0.46445497630331756
    far 0.009478672985781991
    frr 0.5355450236966824

I found LAST configuration :
    conv_1 filters = 23
    conv_2 filters = 179
    conv_3 filters = 37
    Dropout(0.6422912253752693)
    LSTM(22)
    dense_1 = tf.keras.layers.Dense(8, activation = 'relu')(lstm_1)
    dense_2 = tf.keras.layers.Dense(42, activation = 'relu')(dense_1)
    model.compile(loss= 'binary_crossentropy', optimizer= tf.keras.optimizers.SGD(learning_rate=0.0009343048185018842, momentum=0.9379592661454418), metrics=['accuracy'])

I found NEW configuration:
    [[3, 0, 1], 200.0, {"submitted": 1628996905.0295498, "started": 1628996905.0325322, "finished": 1628997832.7254772}, {"loss": 0.06178706884384155, "info": {"test accuracy": 0.9111374616622925, "train accuracy": 0.9790874719619751, "validation accuracy": 0.9382129311561584, "number of parameters": 7344}}, null]
    [[3, 0, 1], {"dropout_rate": 0.27506014989460326, "lr": 0.00287010197769885, "lstm_units": 39, "num_conv_layers": 1, "num_fc_layers": 1, "num_fc_units_1": 11, "num_filters_1": 4, "optimizer": "Adam"}, {"model_based_pick": false}]

    val_loss: 0.1610240489244461
    test_loss: 0.2736242115497589
    test_acc: 0.8909952640533447
    precision: 0.8940397350993378
    recall: 0.6398104265402843
    specificity 0.9747235387045814
    sensitivity :  0.6398104265402843
    far 0.02527646129541864
    frr 0.36018957345971564
    
Sub 6 :
Bohb found configuration
    Best found configuration: {'dropout_rate': 0.16519178106690297, 'lr': 0.0016556065892678969, 'lstm_units': 159, 'num_conv_layers': 2, 'num_fc_layers': 3, 'num_fc_units_1': 8, 'num_filters_1': 48, 'optimizer': 'SGD', 'num_fc_units_2': 241, 'num_fc_units_3': 13, 'num_filters_2': 128, 'sgd_momentum': 0.9102646030327244}
    
    val_loss: 0.08829853683710098
    test_loss: 0.09679172188043594
    test_acc: 0.9644549489021301
    precision: 0.9547738693467337
    recall: 0.9004739336492891
    specificity 0.985781990521327
    sensitivity :  0.9004739336492891
    far 0.014218009478672985
    frr 0.0995260663507109

I found LAST configuration
    conv_1 = tf.keras.layers.Conv1D(filters = 86, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(inputs)
    D_out_1 = tf.keras.layers.Dropout(0.5463367255390843)(max_1)
    lstm_1 = tf.keras.layers.LSTM(154)(D_out_1)
    dense_1 = tf.keras.layers.Dense(33, activation = 'relu')(lstm_1)
    dense_2 = tf.keras.layers.Dense(59, activation = 'relu')(dense_1)
    model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.Adam(0.00025897385811766787), metrics = ['accuracy'])
    
    val_loss: 0.0798117145895958
    test_loss: 0.06874958425760269
    test_acc: 0.9727488160133362
    precision: 0.9234234234234234
    recall: 0.9715639810426541
    specificity 0.9731437598736177
    sensitivity :  0.9715639810426541
    far 0.026856240126382307
    frr 0.02843601895734597

I found NEW configuration



Sub 7 :
Bohb found configuration
    Best found configuration: {'dropout_rate': 0.7947857692503151, 'lr': 0.0007448505160641001, 'lstm_units': 128, 'num_conv_layers': 3, 'num_fc_layers': 2, 'num_fc_units_1': 136, 'num_filters_1': 153, 'optimizer': 'Adam', 'num_fc_units_2': 8, 'num_filters_2': 101, 'num_filters_3': 49}

    val_loss: 0.32515889406204224
    test_loss: 0.49548307061195374
    test_acc: 0.7855450510978699
    precision: 0.5872093023255814
    recall: 0.4786729857819905
    specificity 0.8878357030015798
    sensitivity :  0.4786729857819905
    far 0.11216429699842022
    frr 0.5213270142180095
    
I found LAST configuration :
    conv_1 = tf.keras.layers.Conv1D(filters = 12, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(inputs)
    conv_2 = tf.keras.layers.Conv1D(filters = 12, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(max_1)
    conv_3 = tf.keras.layers.Conv1D(filters = 126, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(max_2)
    D_out_1 = tf.keras.layers.Dropout(0.06330305434316162)(max_3)
    lstm_1 = tf.keras.layers.LSTM(173)(D_out_1)
    dense_1 = tf.keras.layers.Dense(47, activation = 'relu')(lstm_1)
    dense_2 = tf.keras.layers.Dense(140, activation = 'relu')(dense_1)
    dense_3 = tf.keras.layers.Dense(13, activation = 'relu')(dense_2)
    model.compile(loss= 'binary_crossentropy', optimizer= tf.keras.optimizers.SGD(learning_rate=0.09294871404957533, momentum=0.6768319568120289), metrics=['accuracy'])
    
    val_loss: 0.3139367997646332
    test_loss: 0.4419545531272888
    test_acc: 0.8116113543510437
    precision: 0.6830985915492958
    recall: 0.4597156398104265
    specificity 0.9289099526066351
    sensitivity :  0.4597156398104265
    far 0.07109004739336493
    frr 0.5402843601895735
    
I found NEW configuration :
    [[79, 0, 0], 200.0, {"submitted": 1629171896.3666453, "started": 1629171896.3690915, "finished": 1629173489.8361301}, {"loss": 0.10646384954452515, "info": {"test accuracy": 0.8969194293022156, "train accuracy": 0.927122950553894, "validation accuracy": 0.8935361504554749, "number of parameters": 257652}}, null]
    [[79, 0, 0], {"dropout_rate": 0.5502528353650731, "lr": 0.013168480192202132, "lstm_units": 223, "num_conv_layers": 2, "num_fc_layers": 2, "num_fc_units_1": 13, "num_filters_1": 88, "optimizer": "SGD", "num_fc_units_2": 150, "num_filters_2": 45, "sgd_momentum": 0.4245211512655333}, {"model_based_pick": true}]
    
    val_loss: 0.25133731961250305
    test_loss: 0.3119417130947113
    test_acc: 0.8353080749511719
    precision: 0.6935483870967742
    recall: 0.6113744075829384
    specificity 0.909952606635071
    sensitivity :  0.6113744075829384
    far 0.09004739336492891
    frr 0.3886255924170616
    
Sub 10 :

Bohb found configuration
    Best found configuration: {'dropout_rate': 0.5584115172928684, 'lr': 0.0008547152588812913, 'lstm_units': 184, 'num_conv_layers': 2, 'num_fc_layers': 3, 'num_fc_units_1': 67, 'num_filters_1': 6, 'optimizer': 'Adam', 'num_fc_units_2': 193, 'num_fc_units_3': 8, 'num_filters_2': 57}
    
    val_loss: 0.02091953158378601
    test_loss: 0.024042826145887375
    test_acc: 0.9893364906311035
    precision: 0.9855769230769231
    recall: 0.9715639810426541
    specificity 0.995260663507109
    sensitivity :  0.9715639810426541
    far 0.004739336492890996
    frr 0.02843601895734597
    
I found LAST configuration
    conv_1 = tf.keras.layers.Conv1D(filters = 244, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(inputs)
    D_out_1 = tf.keras.layers.Dropout(0.5738648808658326)(max_1)
    lstm_1 = tf.keras.layers.LSTM(212)(D_out_1)
    dense_1 = tf.keras.layers.Dense(103, activation = 'relu')(lstm_1)
    dense_2 = tf.keras.layers.Dense(26, activation = 'relu')(dense_1)
    dense_3 = tf.keras.layers.Dense(8, activation = 'relu')(dense_2)
    model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.Adam(0.0016656074898745027), metrics = ['accuracy'])
    
    val_loss: 0.035076484084129333
    test_loss: 0.02732364647090435
    test_acc: 0.991706132888794
    precision: 0.9766355140186916
    recall: 0.990521327014218
    specificity 0.9921011058451816
    sensitivity :  0.990521327014218
    far 0.007898894154818325
    frr 0.009478672985781991