using GraphTDA
using Test

@testset "GraphTDA.jl" begin
    y = sGTDA()
    @test typeof(y) == sGTDA
    @test GraphTDA.toy() == 5.0
end
