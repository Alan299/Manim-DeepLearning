import numpy as np 
import sys 
sys.path.append('psy')


from manimlib import * 

class Neuron(VGroup):
    CONFIG = {
        'neuron_radius': 1.6,
        'neuron_stroke_color': WHITE,
        'neuron_stroke_width': 8,#3.5,
        'neuron_fill_color': BLUE_C,
        'edge_color': WHITE,
        'edge_stroke_width' : 0.5,
        'edge_propogation_color' : YELLOW,
        'number_inputs':'n',  # integer  or string 'n'
        #'number_outputs': 1, # integer  or string 'n'
        'neuron_spacing' : 3, # factor of separation between node of neuron and inputs outputs.
        


        }

    def __init__(self,single = False, number_outputs = 1,put_label = False,   **kwargs):
        VGroup.__init__(self, **kwargs)

        #self.neuron_radius = neuron_radius
        self.put_label = put_label        
        self.single = single
        self.number_outputs =  number_outputs



        self.logistic = False
        self.node = Circle(
            radius = self.neuron_radius,
            stroke_color = self.neuron_stroke_color,
            stroke_width = self.neuron_stroke_width,
            fill_color = self.neuron_fill_color,
            fill_opacity = 0.9)

        self.label_scale = self.neuron_radius


        self.add(self.node)
        
        self.label =Tex(r"*")
        if self.put_label:
            Tex(r"\sum_{j=0}^{n} \theta_j x_j",
                     tex_to_color_map =  {r'\theta_j': RED_B} )
            self.label_scale = self.neuron_radius - self.neuron_radius/10
            self.label.scale(self.label_scale)
            self.label.move_to(self.node)

            self.activation_label = Tex(r" ")
            self.add(self.label)
            self.add(self.activation_label)

        self.setup_in_out()

    def setup_in_out(self):

        if not self.single:
            return
        text_color =GREY #LIGHT_GREY
        # Setup inputs
        #print("Inputs: ", self.inputs, type(self.inputs))
        if isinstance(self.number_inputs, str):
            self.inputs = VGroup(*[Tex(r'x_0', color =text_color ), Tex(r'x_1',color =text_color),Tex(r'x_2',color =text_color), Tex(r'\vdots',color =text_color), Tex(r'x_n',color =text_color) ])
            self.inputs.scale(2.0)
            self.inputs.arrange(DOWN* self.neuron_spacing)
            self.inputs.next_to(self.node, LEFT * self.neuron_spacing)

            #self.inputs.add_background_rectangle_to_submobjects(color=self.neuron_fill_color, opacity=0.75)

        if isinstance(self.number_inputs, int):
            # x0,x1,x2, . . . , xn | n + 1 inputs
            self.inputs =VGroup(*[Tex(r'x_' + r'{' + str(i) + r'}',color =text_color) for i in range(self.number_inputs + 1 )])
            self.inputs.scale(2.0)
            self.inputs.arrange(DOWN*self.neuron_spacing)
            self.inputs.next_to(self.node, LEFT*self.neuron_spacing)

        """else:
            self.inputs= self.number_inputs"""

        print("Inputs: ", self.number_inputs, type(self.number_inputs))

        # Setup  outputs
        if isinstance(self.number_outputs, str):
            self.outputs =VGroup(*[Tex(r'y_0', color =text_color), Tex(r'y_1',color =text_color),Tex(r'y_2',color =text_color), Tex(r'\vdots',color =text_color), Tex(r'y_n',color =text_color) ])
            self.outputs.scale(3.0)
            self.outputs.arrange(DOWN * self.neuron_spacing)
            self.outputs.next_to(self.node, RIGHT * self.neuron_spacing)

        if isinstance(self.number_outputs, int):

            if self.number_outputs == 1 :
                self.outputs  = VGroup(*[Tex(r'y',color =text_color)])
                self.outputs.scale(2.0)
                self.outputs.next_to(self.node, RIGHT * self.neuron_spacing)

            else:
                # y1,y2, . . . , yn | n  outputs
                self.outputs =VGroup(*[Tex(r'y_' +  r'{'+  str(i) + r'}',color =text_color) for i in range(self.number_inputs + 1 )])
                self.outputs.scale(2.0)
                self.outputs.arrange(DOWN*self.neuron_spacing)
                self.outputs.next_to(self.node, RIGHT*self.neuron_spacing)
        print("Inputs: ", self.inputs, type(self.inputs))
        self.add(self.inputs)
        self.add(self.outputs)
        # Add edges
        self.add_edges()

    def make_logistic_neuron(self):
        # Change opacity to 1
        self.node.fill_opacity = 1.0

        self.logistic = True

        # Change scale
        new_scale = (self.label_scale - 5*self.label_scale/10  )/2
        self.label.scale( new_scale * 1.2 )#1.2



        self.activation_label =VGroup(Tex(r"""g(z) =""",
                                                 tex_to_color_map ={r"z": YELLOW}
                                                ),
                                      Tex(r"""
                                                \frac{1}{1+e^{-z}}""",
                                                tex_to_color_map ={r"z": YELLOW}
                                                )

                                      ).arrange(DOWN* 0.5)

        self.activation_label.scale(new_scale*1.8)

        p1,p2 = self.node.get_center() + np.array([0,self.node.get_height(),0]) ,self.node.get_center() - np.array([0,self.node.get_height(),0])
        self.vertical_line = Line( p1,p2,
            buff = self.neuron_radius,
            stroke_color = self.edge_color,
            stroke_width = self.edge_stroke_width +3,


            )

        self.activation_label.next_to(self.vertical_line, RIGHT)
        self.label.next_to(self.vertical_line, LEFT).shift(RIGHT *0.2)

        self.vertical_line.move_to(self.node)

        self.add(self.vertical_line)

        self.add(self.activation_label)

        self.add(self.label)




    def add_edges(self,add_inputs = True, add_outputs = True):
        self.input_edges = VGroup()
        self.output_edges = VGroup()

        if add_inputs:
            # Add edges to inputs
            for input in self.inputs:
                edge = self.get_edge(input, direction = RIGHT)
                self.input_edges.add(edge)
            #self.add(self.input_edges)
            self.add_to_back(self.input_edges)

        if add_outputs:
            # Add edges to outputs
            if len(self.outputs) == 1:

                output = self.outputs[0]
                output.next_to(self,self.neuron_spacing*RIGHT)
                edge = self.get_edge(output, direction = LEFT,is_arrow = True,use_first_point= True )
                
                self.output_edges.add(edge)

            else:

                for output in self.outputs:
                    #print("Output type: ",type(output),output)
                    edge = self.get_edge(output, direction = LEFT)
                    self.output_edges.add(edge)
            self.add_to_back(self.output_edges)
            #self.add(self.output_edges)





    def get_edge(self, mobject, direction, is_arrow = False,use_first_point  = False ):
        """ Intersection point between this neuron and line how passes from the center of this neuron to one point on the other neuron (mobject or nueron)


        """
        boolean, p1 = self.line_circle_intersection(mobject, direction)

        #print("neuron center and mobject centers: ", self.get_center(), mobject.get_center() )

        if  isinstance(mobject, Neuron) and not use_first_point:
            # If this neuron connects with another neuron. Get intersection between circle and line.
            boolean, p2 =  mobject.line_circle_intersection(self, direction, p1)


        else:
            if use_first_point:
                #use fist point to set second point horizontally until mobject
                
                p2 = p1.copy() 
                p2[0] = (mobject.get_center() -  mobject.get_width()/2)[0]

            else:
                #h = mobject.get_height()
                p2 = mobject.get_boundary_point(direction =direction)
                #p2[1] = p2[1] +  h/2


        assert p1 is not None and p2 is not None, "p1 :{} and p2:{} must not be None".format(p1,p2)


        # shows line creation from left to right
        if p1[0] <= p2[0]:
            point1 = p1
            point2 = p2
        else:
            point1 = p2
            point2 = p1

        if is_arrow:
            return Arrow(
            point1,
            point2,
            stroke_color = self.edge_color,
            stroke_width = self.edge_stroke_width,
            buff=0
            )


        return Line(
            point1,
            point2,
            stroke_color = self.edge_color,
            stroke_width = self.edge_stroke_width,
            )

    # REFERENCE: https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm

    def line_circle_intersection(self, mobject, direction, p2 = None):
        """
        mobject: MObject
            object to connet with this neuron
        direction: 3D numpy array
            Direction from where get bounday point

        Find intersection point bewtween this circle (this neuron node) and segment of line form by center of this neuron and one point in the mobject
        """
        center = self.node.get_center()#center of circle

        # Radius of node. Use width instead of radius to account scaling
        r = self.node.get_width()/2#self.node.radius #self.neuron_radius # radius of circle (node)

        # Points of line segment. line between mobject and center of circle
        p1 = center # Center of circle

        pass_p2 = not p2 is None

        if p2 is None:
            # If not of class Neuron, get middle boundary point
            #h = mobject.get_height()
            p2 = mobject.get_boundary_point(direction =direction)
            #p2[1] = p2[1] -  h/2

        # Vector along line
        v = p2 - p1

        # Compute solutions (coefficients)
        a = v @ v
        b = 2 * v @ (p1-center)
        c = p1 @ p1 + center @ center - 2* p1 @ center  - r**2

        discriminant  = b**2 - 4 * a * c
        if discriminant < 0 :
            return False, None

        sqrt_disc = np.sqrt(discriminant)
        t1 =  (-b + sqrt_disc) / (2 * a)
        t2 =  (-b - sqrt_disc) / (2 * a)


        if not (0 <= t1 <= 1  or 0 <= t2 <= 1):
            return False, None

        #t = max(0, min(1, - b / (2 * a)))

        return True, p1 + t1 * v





    def get_boundary_point(self,direction):

        return self.node.get_boundary_point(direction)

    def arrange(self, **kwargs):
        self.node.arrange(**kwargs)
        self.label.arrange(**kwargs)
        self.activation_label.arrange(**kwargs)


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







class ContinualEdgeUpdate(VGroup):
    CONFIG = {
        "max_stroke_width" : 3,
        "stroke_width_exp" : 7,
        "n_cycles" : 5,
        "colors" : [YELLOW, YELLOW_B, YELLOW_C, RED, PURPLE],
    }
    def __init__(self, network_mob, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.internal_time = 0
        n_cycles = self.n_cycles
        edges = network_mob.get_all_edges( depth= True)
        #VGroup(*it.chain(*network_mob.edge_groups))
        self.move_to_targets = []
        for edge in edges:
            edge.colors = [
                random.choice(self.colors)
                for x in range(n_cycles)
            ]
            msw = self.max_stroke_width
            edge.widths = [
                msw*random.random()**self.stroke_width_exp
                for x in range(n_cycles)
            ]
            edge.cycle_time = 1 + random.random()

            edge.generate_target()
            edge.target.set_stroke(edge.colors[0], edge.widths[0])
            edge.become(edge.target)
            self.move_to_targets.append(edge)
        self.edges = edges
        self.add(edges)
        self.add_updater(lambda m, dt: self.update_edges(dt))

    def update_edges(self, dt):
        self.internal_time += dt
        if self.internal_time < 1:
            alpha = smooth(self.internal_time)
            for move_to_target in self.move_to_targets:
                move_to_target.update(alpha)
            return
        for edge in self.edges:
            t = (self.internal_time-1)/edge.cycle_time
            alpha = ((self.internal_time-1)%edge.cycle_time)/edge.cycle_time
            low_n = int(t)%len(edge.colors)
            high_n = int(t+1)%len(edge.colors)
            color = interpolate_color(edge.colors[low_n], edge.colors[high_n], alpha)
            width = interpolate(edge.widths[low_n], edge.widths[high_n], alpha)
            edge.set_stroke(color, width)



class NeuralNet(VGroup):
    CONFIG= {
        'edge_propogation_color':YELLOW,
        'edge_stroke_width': 0.2,
        'edge_propagation_time' :3.0,
        'height':6,
        'neuron_config':{
            'neuron_radius': 0.2,
            'neuron_stroke_color': GREY,
            'neuron_stroke_width': 0.2,#3.5,
            'neuron_fill_color': BLUE_C,
            'fill_opacity': 0.0,
            'number_inputs':'n',  # integer  or string 'n'
            'neuron_spacing' : 1 # factor of separation between node of neuron and inputs outputs.

        }

    }

    def __init__(self,neurons_per_layer =[3,2,1],spacing =1, **kwargs):
        """
        neurons_per_layer: list of int | None
            if None indicates generic layer xl1 xl2, . . . xlzl

        """

        VGroup.__init__(self,**kwargs)

        #self.number_layers = number_layers      # Does not include ouput layer

        
        self.neurons_per_layer = neurons_per_layer # Include number of inputs, and number of outputs
        
        self.shape = tuple(neurons_per_layer)

        # Input layer

        if self.neurons_per_layer[0]:
            self.inputs = VGroup()
            if self.neurons_per_layer[0] > 30:
                incognite = Tex(r'x')

                for k in range(self.neurons_per_layer[0]):
                    self.inputs.add(
                        incognite.copy()
                )

            else:
                for k in range(self.neurons_per_layer[0]):
                    self.inputs.add(
                        Tex(r'x_{0'+str(k) + r'}')
                    )

        else:
            #Generic layer
            self.inputs = VGroup(*[Tex(r'x_{0}'), Tex(r'x_{1}'),Tex(r'x_{2}'), Tex(r'\vdots'), Tex(r'x_{n}') ])
           


        #self.inputs.scale(2.0)
        self.inputs.arrange(DOWN)
        #self.inputs.to_edge(LEFT)
        self.add(self.inputs)


        # Output layer
        if self.neurons_per_layer[-1]:

            self.outputs = VGroup(*[
                Tex(r'y_'+r'{'+str(k) + r'}') for k in range (self.neurons_per_layer[-1])
                
            ])
        else:
            self.outputs= VGroup(*[Tex(r'y_0'), Tex(r'y_1'),Tex(r'y_2'), Tex(r'\vdots'), Tex(r'y_{n}') ])
        
        #self.outputs.scale(2.0)
        self.outputs.arrange(DOWN)


        self.layers = VGroup()
        self.spacing = spacing
        # Add neurons for each layer
        for i in range(1,len(self.neurons_per_layer)):
            neurons = VGroup()
            # USE MAP(func,iter) to set neurons
            for j   in range(self.neurons_per_layer[i]):
                if i == len(self.neurons_per_layer) -1:
                    # If last layer
                    neuron = Neuron(number_ouputs =self.neurons_per_layer[-1],
                    **self.neuron_config )
                else:
                    neuron = Neuron(number_ouputs =self.neurons_per_layer[i+1],
                    **self.neuron_config )

                neurons.add(neuron)

            neurons.arrange(DOWN)

            self.layers.add(neurons)


        self.edges = VGroup()

        self.layers.arrange(RIGHT * self.spacing)
        self.add(self.layers)

        self.layers.next_to(self.inputs, RIGHT * self.spacing * 1.2)

        #RIGHT * self.layers[-1][0].neuron_spacing
        self.outputs.next_to(self.layers[-1],RIGHT * self.spacing * 0.5 )
        self.add(self.outputs)




        #self.connect()

        #self.set_height(self.height)

    
    def connect(self):
        # Setup inputs outputs for each neuron

        print("len neurons per layer : ", len(self.neurons_per_layer),len(self.layers))
        # Setup inputs and ouputs
        last_layer = self.inputs
        for i in range(len(self.layers)):
            print("i", i)
            for j in range( len(self.layers[i])):

               
                neuron = self.layers[i][j]

                 # Set inputs
                neuron.inputs = last_layer

                # Set outputs
                if i == len(self.layers) - 1:
                    neuron.outputs = self.outputs

                else:
                    neuron.outputs = self.layers[i+1]

            last_layer = self.layers[i]


        # Add EDGES
        # Add only one time shared edges. Example between layer 1 and 2 only add output edges of layer 1
        for i in range(len(self.layers)):
            neurons_i = VGroup()
            for j in range(len(self.layers[i])):
                neuron = self.layers[i][j]

                if i == len(self.layers) -1:
                    neuron.add_edges(add_inputs = True, add_outputs = True)

                else:
                    neuron.add_edges(add_inputs = True, add_outputs = False)


    def get_all_edges(self, depth= False):
        """
        Get all the edges from the net
        depth: Boolean 
            if True gives a vgroup where each element is a edge (not other group)
        """
        if depth:
            
            n=  len(self.layers)
            edges = []

            for k in range(n):
                edges+= list(
                    # Edges are gruop by neuron, unpack them.
                    it.chain(*self.get_edges_from_layer(k)) 
                    )
            all_edges = VGroup(
                *edges 

            )

        else:
            all_edges = VGroup(   
                *it.chain(
                    *(self.get_edges_from_layer(i) for i in range(len(self.layers)))
                    )
                )

        return all_edges

        

            


    def get_edges_from_layer(self,index):
        """
        Get edges from the layer given by index

        """
        # get all edges from a layer
        edge_group_copy =VGroup(
            *list(map(lambda  d: self.get_edges(d,index) ,self.layers[index] ) )
            
        )
        return edge_group_copy

    def get_number_edges(self):
        inner_layers = sum(  len(self.layers[k]) * len(self.layers[k+1]) 
            for k in range(len(self.layers) - 1)
            )
        return inner_layers  + len(self.inputs) * len(self.layers[0])
    
    def get_number_params(self):
        n = (self.get_number_edges()
            + sum(len(self.layers[j]) for j in range(len(self.layers)))
        )
        return n

    def get_edges(self,neuron, index):
        """
        Get edges from the given neuron 
        """    
        edges  = VGroup()

        input_edges = neuron.input_edges.copy()
        

        if index == len(self.layers) - 1:
            output_edges = neuron.output_edges.copy()
            edges.add(
                *it.chain(
                    *(output_edges, input_edges)

                    )
                )
        else:
            edges.add(*input_edges)

        return edges

    def get_edge_propogation_animations(self,index,width_scale= 2.5 ):
        
        """def get_input_edges(neuron):
            return neuron.input_edges


        def get_output_edges(neuron):
            return neuron.output_edges

        """
        self.index = index

        # get all edges from a layer
        edge_group_copy =self.get_edges_from_layer(index)
        

        edge_group_copy.set_stroke(
            self.edge_propogation_color,
            width = width_scale *self.edge_stroke_width
        )

        n_edges = len(edge_group_copy)
        print("N edges: ", n_edges)
        #usepropagation time given by number of edges
        edge_propogation_time_ =  0 

        return [ShowCreationThenDestruction(
            edge_group_copy, 
            run_time = self.edge_propagation_time,
            lag_ratio = 0.2
        )]
    def animate(self):
        pass



class BlackBoxModel(VGroup):
    CONFIG = {
        "title":r"Modelo",
        "title_config":{
            'color': BLUE,
            "stroke_color":YELLOW_C ,
            "stroke_opacity": 1.0,
            "stroke_width": DEFAULT_STROKE_WIDTH,
            },
        'height':3,
        'box_config':
            {
                'n': 4,
                'color':BLUE_C,
                "fill_opacity": 0.45,
                "stroke_color":GREY , #LIGHT_GRAY,
                "stroke_opacity": 1.0,
                "stroke_width": DEFAULT_STROKE_WIDTH*1.5,
            },
        

    }

    def __init__(self,**kwargs):
        VGroup.__init__(self,**kwargs)
        
        
        self.box = RegularPolygon(**self.box_config)
        self.box.rotate(PI/4)

        self.model_title  = Text(self.title, **self.title_config)
        self.model_title.set_width(self.box.get_width() * 0.9)


        self.add(self.box)
        self.add(self.model_title)
        self.set_height(self.height)





