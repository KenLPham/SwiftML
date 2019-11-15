# SwiftML

[![CI Status](https://img.shields.io/travis/KenLPham/SwiftML.svg?style=flat)](https://travis-ci.org/KenLPham/SwiftML)
[![Version](https://img.shields.io/cocoapods/v/SwiftML.svg?style=flat)](https://cocoapods.org/pods/SwiftML)
[![License](https://img.shields.io/cocoapods/l/SwiftML.svg?style=flat)](https://cocoapods.org/pods/SwiftML)
[![Platform](https://img.shields.io/cocoapods/p/SwiftML.svg?style=flat)](https://cocoapods.org/pods/SwiftML)
[![Twitter](https://img.shields.io/twitter/follow/lilboipham?label=Ken%20Pham&style=social)](https://twitter.com/lilboipham)

## Description

Create and train a neural network on device. Import your trained model from TensorFlow and create your model in one line. Once your model is imported, you can input values into the model and get the outputs. If you would like to personalize the model to the user, you can call a method and provide the output and the expected results, and the framework will use Backpropagation to train your model.

*I'm new to this whole machine learning thing. This code mainly available for educational purposes. If there are any errors or anything you would like to see added please make an Issue!*

## To Do

- Add other alogrithms
- Implement other connection types (currently only supports fully connected layers)
- Optimize (Move out of UI thread, fix first node check)

## Installation

SwiftML is available through [CocoaPods](https://cocoapods.org). To install
it, simply add the following lines to your Podfile:

```ruby
source 'https://github.com/CocoaPods/Specs.git'
source 'https://github.com/KenLPham/Spec.git'
platform :ios, '11.0'
use_frameworks!

target 'MyApp' do
  # your other pod
  # ...
  pod 'PT+SwiftML'
end
```

## Usage



## Author

Ken Pham

## License

MIT License. See the LICENSE file for more info.
