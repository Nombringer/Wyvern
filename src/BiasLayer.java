

/**
 * Created by fabd on 12/09/17.
 */
  class BiasLayer extends Layer {

    private ActivationFunction layerActivationFunction;
    private Neuron[] neurons;

    public BiasLayer(int i, int i1, ActivationFunction layerFunction) {
        super(i, i1, layerFunction);

        neurons = new Neuron[this.getColumnDimension()];
        for (int j = 0; j < neurons.length; j++) {
            neurons[i] = new Neuron(i, layerFunction);
        }
    }


    public void activate(){
        for (int i = 0; i < this.getRowDimension(); i++) {
            for (int j = 0; j < this.getColumnDimension(); j++) {
                this.set(i, j, neurons[j].getActivationFunction().apply(this.get(i, j)));
            }
        }
    }


    public void setActivationFunction(int index, ActivationFunction func){
        neurons[index].setActivationFunction(func);
    }

    public ActivationFunction getActivationFunction(int index){
        return neurons[index].getActivationFunction();
    }

    //TODO: Maybe this implementation is a little ugly. Find a way to make it work or just din't use the inner class.
    private class Neuron {

        private ActivationFunction activationFunction;

        Neuron(int pos, ActivationFunction function){
            activationFunction = function;
        }

         ActivationFunction getActivationFunction() {return activationFunction;}

         void setActivationFunction(ActivationFunction activationFunction) {
            this.activationFunction = activationFunction;}
    }

}