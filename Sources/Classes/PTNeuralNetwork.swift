//
//  PTNeuralNetwork.swift
//  MobileMind
//
//  Created by Ken Pham on 6/22/19.
//  Copyright © 2019 PhezTech. All rights reserved.
//

import Foundation
import Accelerate

/** Trainable Neural Network using Back Propagation
 
 DISCLAIMER: This is the first time I've ever worked with anything Machine Learning related. This setup probably has a lot of issues/not how a neural network should be written. It just works for my uses, so I'm assuming its somewhat valid.
 
 At the moment, these layers are all fully connected.
 
 Resources:
 - https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
 - https://stackoverflow.com/questions/3775032/how-to-update-the-bias-in-neural-network-backpropagation
 - https://stackoverflow.com/questions/56465742/generic-simd-math
 - https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/neural_networks.html
 - http://cs231n.github.io/convolutional-networks/#conv
 
 
 Other Topics
 - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
 - https://machinethink.net/blog/recurrent-neural-networks-with-swift/
 - https://medium.com/datathings/neural-networks-and-backpropagation-explained-in-a-simple-way-f540a3611f5e
 - https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464
 - https://www.spss-tutorials.com/pearson-correlation-coefficient/
 - https://towardsdatascience.com/neat-an-awesome-approach-to-neuroevolution-3eca5cc7930f
 
 Learning
 - https://machinelearningmastery.com/start-here/
 - https://machinelearningmastery.com/k-fold-cross-validation/
 - https://machinelearningmastery.com/python-machine-learning-mini-course/
 
 Training Methods
 - https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3?gi=858505a3bc21
 
 - todo: pruning
 */

/**
 */
open class PTNeuralNetwork: NSObject {
    public typealias Input = [Float]
    public typealias Weights = [Float]
    public typealias Bias = [Float]
    public typealias Output = [Float]
    
    private var allWeights: [Weights]!
    private var allBias: [Bias]!
    
    private var function: PTActivation.Function!
    private var partial: PTActivation.Function!
    
    private var size: [vDSP_Length]!
    
    /// Learning Rate
    private var n: [Float] = [0.5]
    
    /// He Initialization (Method to create initial weights)
    public class func he (size s: Int) -> Float {
        return Float.random(in: 0...1) * sqrt(2/Float(s))
    }
    
    /// Xavier Initialization (Only use for tanh)
    public class func xavier (size s: Float) -> Float {
        return Float.random(in: 0...1) * sqrt(1/s)
    }
    
    /// - todo: figure out what i meant by implement
    /// - todo: implement
    convenience init (size s: [Int], bias b: [Bias], activation f: PTActivationFunction) {
        var aW: [Weights] = []
        
        for i in 1..<(s.count-1) {
            let row = s[i]
            let col = s[i+1]
            
            let mtrx = row * col
            
            var weights: Weights = []
            
            for _ in 0..<mtrx {
                weights.append(PTNeuralNetwork.he(size: row))
            }
            
            aW.append(weights)
        }
        
        let matrix: [vDSP_Length] = s.map({ vDSP_Length($0) })
        
        self.init(weights: aW, bias: b, size: matrix, activation: f)
    }
    
    /** Single precision Neural Network with Back propagation training
     - todo: Throw error if w.count and b.count arent equal
     
     - Parameters:
     - w: Array of weights for each layer
     - b: Array of biases for each layer
     - s: The number of neurons for each layer
     */
    public convenience init (weights w: [Weights], bias b: [Bias], size s: [vDSP_Length], activation f: PTActivationFunction, transpose: Bool = true) {
        self.init()
        
        // Transpose weights from MxN to NxM
        if transpose {
            self.allWeights = []
            for x in 0..<w.count {
                var transposed = w[x]
                vDSP_mtrans(transposed, 1, &transposed, 1, s[x+1], s[x])
                
                self.allWeights.append(transposed)
            }
        } else {
            self.allWeights = w
        }
        
        self.allBias = b
        self.function = PTActivation.function(f)
        self.partial = PTActivation.partial(f)
        
        self.size = s
    }
    
    private override init () {
        super.init()
    }
    
    /** Set the learning rate of the network
     The learning rate has to be greater than 1e-5. The lower it is, the longer it will take for the network to train, but it will result in a more accurate network
     
     - Parameter n: The new learning rate you want to set (default: 0.5)
     */
    public func setRate (to n: Float) {
        self.n[0] = n
    }
    
    /** Forward pass through the whole neural network
     This function will return the output of the final layer. If you would like to dynamically train your model, use the forward(input:) method
     
     - Parameter i: Input of the neural network
     - Returns: Output of the final layer
     */
    public func apply (input i: Input) -> Output {
        return self.forward(input: i).last!
    }
    
    /** Forward pass through the whole neural network
     This function will return the output of each layer. This is needed if you would like to dynamically train the neural network (using reinforce). If your model is already trained, and you don't plan on training it any further, use the apply(input:) method.
     
     - Parameter i: Input of the neural network
     - Returns: Array of the output of each layer
     */
    public func forward (input i: Input) -> [Output] {
        var outputs: [Output] = []
        
        for j in 0..<self.allWeights.count {
            let layer = (j == 0) ? i : outputs[j-1]
            let output = self.forward(input: layer, weights: self.allWeights[j], bias: self.allBias[j])
            
            outputs.append(output)
        }
        
        return outputs
    }
    
    /** Forward Pass through a single layer of the neural network
     - Parameters:
     - i: Input of the layer
     - w: Weights of the current layer
     - b: Bias of the current layer
     - Returns: Output of the layer
     */
    private func forward (input i: Input, weights w: Weights, bias b: Bias) -> Output {
        let count = b.count
        let bLen = vDSP_Length(count)
        let iLen = vDSP_Length(i.count)
        
        var result = [Float](repeating: .nan, count: count)
        
        vDSP_mmul(w, 1, i, 1, &result, 1, bLen, 1, iLen)
        vDSP_vadd(result, 1, b, 1, &result, 1, bLen)
        
        return self.function(result)
    }
    
    /** Supervised training of the neural network
     - Parameters:
     - data: Set of data used to train the model
     - expected: The expected output of each input from the training data
     - error: The minimum error you want the model to have (Default: 1e-6)
     */
    public func train (data trainingSet: [Input], expected e: [Output], error min: Float = 1e-6) {
        var passes = 0
        var totalError: Float = 2
        
        let data = zip(trainingSet, e)
        while totalError > min {
            for (set, ex) in data {
                // forward pass
                let out = self.forward(input: set)
                
                // backwards pass
                totalError = self.reinforce(input: set, results: out, expected: ex)
                
                passes += 1 // iterations
            }
        }
    }
    
    /** Supervised training of the neural network
     - Parameters:
     - data: Set of data used to train the model
     - expected: The expected output of each input from the training data
     - epoch: The number of forward and backward passes of the all the training data
     */
    @discardableResult
    public func train (data trainingSet: [Input], expected e: [Output], epoch max: Int) -> Float {
        var epoch = 0
        var totalError: Float = 1
        
        let data = zip(trainingSet, e)
        while epoch < max {
            for (set, ex) in data {
                // forward pass
                let out = self.forward(input: set)
                
                // backwards pass
                totalError = self.reinforce(input: set, results: out, expected: ex)
            }
            epoch += 1
        }
        
        return totalError
    }
    
    /** Dynamic supervised training of the neural network
     Use this method to continue training your neural network with real data. This is still supervised training so you will have to have the expected result (e.g: obtained through user input)
     
     - Parameters:
     - i: Input of the neural network
     - r: Results of the neural network with provided input
     - e: Expected output with the provided input
     - Returns: Error of the output from the expected value
     */
    @discardableResult
    public func reinforce (input i: Input, results r: [Output], expected e: Output) -> Float {
        let totalError = self.error(expected: e, output: r.last!)
        
        // backwards pass
        let new = self.backprop(input: i, results: r, expected: e)
        
        // update and save weights
        self.save(new.weights, new.bias)
        
        return totalError
    }
    
    private func save (_ weights: [Weights], _ bias: [Bias]) {
        self.allWeights = weights
        self.allBias = bias
    }
    
    private func backprop (input ip: Input, results r: [Output], expected e: Output) -> (weights: [Weights], bias: [Bias]) {
        let rCount = r.count
        
        var prevCost: [[Float]] = [[Float]](repeating: [], count: rCount)
        var newWeights = [Weights](repeating: [], count: rCount)
        var newBias = [Bias](repeating: [], count: rCount)
        
        for i in stride(from: (r.count-1), through: 0, by: -1) {
            let layer = r[i]
            let lCount = layer.count
            let lLen = vDSP_Length(lCount)
            
            let previous = (i == 0) ? ip : r[i-1] // layer output
            let pLen = vDSP_Length(previous.count)
            
            let dsdl = self.partial(layer)
            
            let curWeights = self.allWeights[i]
            let cWCount = curWeights.count
            let cWLen = vDSP_Length(cWCount)
            
            let curBias = self.allBias[i]
            let cBCount = curBias.count
            let cBLen = vDSP_Length(cBCount)
            
            if i == (r.count - 1) { // output
                // Error
                let eCount = e.count
                let eLen = vDSP_Length(eCount)
                
                var loss = [Float](repeating: .nan, count: eCount)
                vDSP_vsub(e, 1, layer, 1, &loss, 1, eLen)
                vDSP_vmul(loss, 1, dsdl, 1, &loss, 1, eLen)
                
                // Weights
                var wO = Weights(repeating: .nan, count: cWCount)
                vDSP_mmul(previous, 1, loss, 1, &wO, 1, pLen, lLen, 1)
                vDSP_vsmul(wO, 1, self.n, &wO, 1, cWLen)
                vDSP_vsub(wO, 1, curWeights, 1, &wO, 1, cWLen)
                
                // Bias
                var bO = Bias(repeating: .nan, count: cBCount)
                vDSP_vsmul(loss, 1, self.n, &bO, 1, cBLen)
                vDSP_vsub(bO, 1, curBias, 1, &bO, 1, cBLen)
                
                // Save
                prevCost[i] = loss
                newWeights[i] = wO
                newBias[i] = bO
            } else {
                let preC = prevCost[i+1]
                let cLen = vDSP_Length(preC.count)
                
                let preW = self.allWeights[i+1] // lLen * cBLen
                
                // Error
                var error = [Float](repeating: .nan, count: lCount)
                vDSP_mmul(preW, 1, preC, 1, &error, 1, cBLen, 1, cLen)
                vDSP_vmul(error, 1, dsdl, 1, &error, 1, lLen)
                
                // Weights
                var weights = Weights(repeating: .nan, count: cWCount)
                vDSP_mmul(error, 1, previous, 1, &weights, 1, cBLen, pLen, 1)
                vDSP_vsmul(weights, 1, self.n, &weights, 1, cWLen)
                vDSP_vsub(weights, 1, curWeights, 1, &weights, 1, cWLen)
                
                // Bias
                var bias = Bias(repeating: .nan, count: cBCount)
                vDSP_vsmul(error, 1, self.n, &bias, 1, cBLen)
                vDSP_vsub(error, 1, curBias, 1, &bias, 1, cBLen)
                
                // Save
                prevCost[i] = error
                newWeights[i] = weights
                newBias[i] = bias
            }
        }
        
        return (newWeights, newBias)
    }
    
    private func error (expected e: Output, output o: Output) -> Float {
        var results = [Float](repeating: .nan, count: e.count)
        let count = vDSP_Length(e.count)
        
        vDSP_vsub(e, 1, o, 1, &results, 1, count)
        vDSP_vsq(results, 1, &results, 1, count)
        vDSP_vsmul(results, 1, [0.5], &results, 1, count)
        
        var sum: Float = .nan
        vDSP_sve(results, 1, &sum, count)
        
        return sum
    }
}
