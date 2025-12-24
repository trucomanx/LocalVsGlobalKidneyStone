# Definições do modelo

import tensorflow as tf


MODEL_MAP = {
    'EfficientNetV2S': tf.keras.applications.efficientnet_v2.EfficientNetV2S,
    'EfficientNetV2M': tf.keras.applications.efficientnet_v2.EfficientNetV2M
# Adicionar outros modelos AQUI e lembrar de trocar o nome do usado atualmente
}

MODEL_MAP_SIZE =  {
    'EfficientNetV2S': (384, 384),
    'EfficientNetV2M': (480, 480)
# Adicionar outros modelos AQUI e lembrar de trocar o nome do usado atualmente
}


def build_model(model_name='EfficientNetV2S', dropout=0.3):
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model_name: {model_name}. Available: {list(MODEL_MAP.keys())}")
    backbone_cls = MODEL_MAP[model_name]
    input_shape = (MODEL_MAP_SIZE[model_name][0], MODEL_MAP_SIZE[model_name][1], 3)
    backbone = backbone_cls(include_top=False, input_shape=input_shape, weights='imagenet', pooling='avg')

    inputs = tf.keras.Input(shape=input_shape)
    x = backbone(inputs, training=False)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs, name=f"{model_name}_bincls")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
    )
    return model, MODEL_MAP_SIZE[model_name]

print('Definição do modelo pronta')
