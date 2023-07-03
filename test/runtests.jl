using GraphTDA
using Test

@testset "GraphTDA.jl" begin
    y = sGTDA()
    @test typeof(y) == sGTDA
    @test toy() == 5.0
end
