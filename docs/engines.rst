**Engines**
===============

To aid in lowering the barrier of using **n2v**, we introduce the use of engines. 
We call an engine any computational chemistry code that can be interfaced with **n2v** to perform its tasks. 
This is done through an abstract class called the *Engine*. This class defines all the methods that an engine needs to perform to function properly with **n2v**.
The *Engine* expects each package to be able to produce components such as matrices in the atomic orbital basis as well as components on the grid.

In order to use or extend the Engines framework, one ought to work in the *Engine* branch of n2v. 
All engines are located in the *n2v/engines* folder. The file *engine.py* contains an abstract class that specifies an engine's methods. The currently available methods require these. If one is only interested in one or a few methods, one can always add the method with a *pass* inside to make Python happy. 

Most methods are standard and straightforward, like requesting an overlap matrix of the AO basis set used or computing the Hartree potential/energy given an electronic density in the AO basis. 

Next, one needs to add the new engine to the __init__ in the Inverter located in *n2v/inverter.py*. 