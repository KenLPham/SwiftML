Pod::Spec.new do |s|
    s.module_name       = "SwiftML"
    s.name              = "PT+SwiftML"
    s.version           = "0.1"
    s.summary           = "Create and train neural networks on device"
    s.homepage          = "https://github.com/pheztech/"
    s.license           = { :type => "MIT", :file => "LICENSE.md" }
    s.author            = { "Ken Pham" => "ken@pheztech.com" }
    s.source            = { :git => "https://github.com/pheztech/SwiftML.git", :tag => s.version }
    s.social_media_url  = 'https://twitter.com/lilboipham'
    
    s.platform          = :ios, '11.0'
    s.requires_arc      = true
    s.swift_versions    = '5.0'
    
    s.source_files      = 'Sources/Classes/**/*.swift'
    
    s.frameworks        = 'Accelerate'
end

