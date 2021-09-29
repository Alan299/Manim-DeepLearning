import sys 
sys.path.append('./..')
sys.path.append('.')


from manimlib import * 
from deep_learning import * 




class Intro(Scene):


    def construct(self):

        """
        add_foreground_mobjects().  bring_to_front() and bring_to_back().

        """
        self.setup()
        self.animate_neuron(self.lr_neuron)

        # Clean neuron
        self.clear()
        self.wait(3)

        self.animate_neuron(self.logistic_neuron)

        # Clear
        self.clear()

        self.animate_neural_net(self.nn)

    
    def setup(self):
        self.lr_neuron = Neuron(single = True)

        self.logistic_neuron  = Neuron(neuron_radius = 2.2,put_label=True, single = True)
        self.logistic_neuron.CONFIG['neuron_spacing'] = 4.0
        self.logistic_neuron.make_logistic_neuron()

        self.nn = NeuralNet(number_layers = 1,number_neuron =[3,1],spacing = 6,height = 24)
        self.nn.move_to(ORIGIN )
        self.nn.connect()




    def animate_neuron(self,neuron):

        # Create node
        self.play(ShowCreation(neuron.node, run_time = 5))

        #self.add_sound('./assets/roblox.wav')

        # Animate inputs
        self.play(LaggedStartMap(FadeIn, neuron.inputs,lag_ratio = 9.0 ))
        self.play(ShowCreation(neuron.input_edges))


        # Animate outputs
        self.play(ShowCreation(neuron.output_edges))
        self.play(LaggedStartMap(FadeIn, neuron.outputs ))


        # Add label
        if neuron.logistic:
            self.play(ShowCreation(neuron.vertical_line, run_time = 5))

            neuron.label.next_to(neuron.vertical_line, LEFT)
            self.play(Write(neuron.label, run_time = 3))

            self.wait(2)
            neuron.activation_label.next_to(neuron.vertical_line, RIGHT)
            self.play(Write(neuron.activation_label, run_time = 3))


        else:
            self.play(Write(neuron.label, run_time = 3))


        self.wait(6)


    def animate_neural_net(self,nn):

        # Animate inputs
        self.play(
        LaggedStartMap(
            FadeIn,nn.inputs,
            #run_time = 6,
            lag_ration = 0.9

         ))


        for  k in range(len(nn.layers)):
            l = nn.layers[k]

            self.play(LaggedStartMap( ShowCreation, VGroup( *list(map(lambda d: d.node , l ))), run_time = 5 ))
            self.wait(3)

            self.play(LaggedStartMap( ShowCreation, VGroup( *list(map(lambda d: d.input_edges,l )))   , run_time = 5  ))
            self.wait(2)


        # Animate outputs
        self.play(LaggedStartMap( ShowCreation,VGroup(*list(map(lambda d: d.output_edges,nn.layers[-1] )))   , run_time = 2  ))
        self.play(
        LaggedStartMap(
            ShowCreation,nn.outputs,
            run_time = 3

            ))
        self.wait(5)
