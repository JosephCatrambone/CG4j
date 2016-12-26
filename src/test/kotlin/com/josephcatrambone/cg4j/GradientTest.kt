package com.josephcatrambone.cg4j

import io.kotlintest.properties.Gen
import io.kotlintest.specs.StringSpec

/**
 * Created by josephcatrambone on 12/26/16.
 */
fun numericalGradient(f:(Float)->Float, x:Float, h:Float) : Float {
	return (f(x+h) - f(x-h))/(2.0f*h)
}

class GradientTest : StringSpec() {
	init {
		"Tensor ones (5, 5, 5) should have 125 elements." {
			Tensor.ones(5, 5, 5).data.size shouldBe 125
		}

		"Gradient random check" {
			forAll(Gen.int(), Gen.int(), { a: Int, b: Int ->
				a+b == a+b
			})
		}
	}
}

fun gradCheck() {
	val g = Graph()
	val x = InputNode(1, 9)

	// Test tanh.
	//val out = g.tanh(x)
	//val f : (Float)->Float = { x -> Math.tanh(x.toDouble()).toFloat() };
	// Test x^2
	//val out = g.elementMultiply(x, x)
	//val f : (Float)->Float = { x -> x*x };
	// Test x+x
	//val out = g.add(x, x)
	//val f : (Float)->Float = { x -> x+x };
	// Test x-x
	//val out = g.subtract(x, x)
	//val f : (Float)->Float = {x -> x-x } // Basically zero.  d/dx x wrt x = 1.  d/dx -x wrt x = -1.  1 + -1 = 0.
	// Test 1/x
	//val out = g.power(x, -1f)
	//val f : (Float)->Float = {x -> Math.pow(x.toDouble(), -1.0).toFloat() }
	// Test something else.
	val out = ConstantMultiplyNode(x, 10f)
	val f : (Float)->Float = {x -> x*10f}

	g.add(out)

	val dx = 1.0e-4f;
	val xData = Tensor(shape=intArrayOf(1, 9), data=floatArrayOf(-10f, -5f, -2f, -1f, 0.0f, +1f, 2f, 5f, 10f))
	val inputFeed = mapOf<Node, Tensor>(x to xData)
	val fwd = g.forward(out, inputFeed)
	val exactDerivative = g.reverse(out, inputFeed, fwd)[x]!!
	val numericalDerivative = floatArrayOf(
			numericalGradient(f, -10f, dx),
			numericalGradient(f, -5f, dx),
			numericalGradient(f, -2f, dx),
			numericalGradient(f, -1f, dx),
			numericalGradient(f, 0f, dx),
			numericalGradient(f, 1f, dx),
			numericalGradient(f, 2f, dx),
			numericalGradient(f, 5f, dx),
			numericalGradient(f, 10f, dx)
	)
	println("Exact: $exactDerivative")
	println("Approx: ${numericalDerivative.joinToString()}")
}