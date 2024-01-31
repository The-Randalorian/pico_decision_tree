# Pico Decision Tree
A small decision tree machine learning library written in C++. Designed to have only the necessary features for a decision tree. This should be good for microcontrollers like the Raspberry Pi Pico (though you might want to make some optimizations for your specific microcontroller.) This library is designed to classify data from sensors on a microcontroller into discrete buckets or values.

For the purposes of this readme, "good for microcontrollers" means "not designed with a ton of unnecessary bells and whistles." It does not mean this is designed to run on every tiny, low spec MCU. It doesn't make decisions like "what's the best int size for this data type?" or "are floats enough precision?" and the like. When deploying this to your own device, I highly recommend adjusting it and tailoring it to the specific hardware and application.

## Features
* Decision Tree Fitting - Fit a decision tree to a given data set.
* Decision Tree Prediction - Classify a sample.
* Decision Tree (De-)Serialization - Convert a decision tree into data then back into a tree. Useful to save/load a decision tree to/from persistent storage.

## Tree Structure
Currently, this library only handles decision trees with continuous inputs and discrete outputs. It is possible to create decision trees with discrete inputs and discrete outputs by only sending discrete inputs to the continuous inputs, but these trees will still be processed as trees with continous inputs.

This structure was picked to simplify the logic of the library. As the focus of this library is around classifying continous sensor data into discrete output buckets, this library was optimized for the continuous-input discrete-ouput case.

## Serialized Data Structure
The serialized data structure created by this library should not be considered a standard format, and is not designed to be interoperable with other programs ***or even other architectures;*** instead, it is designed to store the decision tree information in a format as close to the final decision tree as possible, minus most unnecessary information. ***This includes using the processor default byte ordering and data size.*** The data is stored similar to postfix notation. As the data is read from start to end, the final decision tree is created by adding and removing trees from a tree stack. The pseudocode of this system is as follows:

1. Read byte.
2. Is byte 0xAA?
    1. Read default_value. (int)
    2. Add DecisionTreeNode with default_value to top of stack.
    3. Go to start.
3. Is byte 0xBB?
    1. Read compare_parameter. (size_t)
    2. Read compare_threshold. (double)
    3. Pop greater_branch from top of stack.
    4. Pop lesser_branch from top of stack.
    5. Add DecisionTreeNode with compare_parameter, compare_threshold, greater_branch, and lesser_branch to top of stack.
    6. Go to start.

The values 0xAA and 0XBB are used pretty arbitrarily; they are only used because they are easy to distinguish when looking at hex directly. ***The specific length and byte ordering of the default_value, compare_parameter, and compare_threshold fields may change depending on the processor. It is not recommended to send decision trees made by one processor to another.***

The data structure format ***does not*** encode the length, number of parameters, or number of labels. The number of parameters and labels should normally be in your program as constant values though, but worst case you can handle storing that information yourself. You must save the length on your own though, as the behavior of the deserialization functionality is undefined when the length given does not match the length of the actual serialized decision tree. Saving the length right in front of the byte array in persistent storage would be sufficient.
