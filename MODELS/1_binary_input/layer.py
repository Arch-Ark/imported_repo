from tensorflow.keras.models import load_model

# Load the model
model = load_model('model0.keras')

# Get the output shape of the final layer
#output_shape = model.layers[-1].output_shape
output_shape = model.layers[-2]

print(output_shape.units)

# Extract the number of classes
#num_classes = output_shape[1] if len(output_shape) == 2 else output_shape[-1]

#print(f'The model has {num_classes} classes.')
