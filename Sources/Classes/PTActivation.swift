//
//  PTActivation.swift
//  SwiftML
//
//  Created by Ken Pham on 6/21/19.
//  Copyright © 2019 PhezTech. All rights reserved.
//

import Foundation
import Accelerate
import simd

public enum PTActivationFunction {
    case sigmoid, relu, leakyRelu
}

public struct PTActivation {
    public typealias Function = ([Float])->([Float])
    public typealias FunctionD = ([Double])->([Double])
    
    public static var leakyRelu: Function = { x in
        return x.map({ ($0 >= 0) ? $0 : $0*0.2 })
    }
    
    public static var partialLeakyRelu: Function = { x in
        return x.map({ ($0 >= 0) ? 1 : 0.2 })
    }
    
    // Double Percision
    public static var leakyReluD: FunctionD = { x in
        return x.map({ ($0 > 0) ? $0 : $0*1e-4 })
    }
    
    public static var partialLeakyReluD: FunctionD = { x in
        return x.map({ ($0 > 0) ? 1 : 1e-4 })
    }
    
    /// - todo: try using Accelerate instead (this is available with the new Swift vDSP I think)
    public static var relu: Function = { x in
        return x.map({ max(0, $0) })
    }
    
    public static var partialRelu: Function = { x in
        return x.map({ min(max(0, $0), 1) })
    }
    
    public static var sigmoid: Function = { x in
        return x.map({ 1/(1+exp(-$0)) })
    }
    
    public static var partialSigmoid: Function = { x in
        var result = [Float](repeating: .nan, count: x.count)
        let y = [Float](repeating: 1, count: x.count) // bruh
        let count = vDSP_Length(x.count)
        vDSP_vsub(x, 1, y, 1, &result, 1, count)
        vDSP_vmul(x, 1, result, 1, &result, 1, count)
        
        return result
    }
    
    public static func function (_ function: PTActivationFunction) -> Function {
        switch function {
        case .relu:
            return relu
        case .leakyRelu:
            return leakyRelu
        default:
            return sigmoid
        }
    }
    
    public static func partial (_ function: PTActivationFunction) -> Function {
        switch function {
        case .relu:
            return partialRelu
        case .leakyRelu:
            return partialLeakyRelu
        default:
            return partialSigmoid
        }
    }
}
