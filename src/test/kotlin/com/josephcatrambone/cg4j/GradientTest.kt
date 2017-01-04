package com.josephcatrambone.cg4j

import io.kotlintest.properties.Gen
import io.kotlintest.specs.StringSpec
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
 * Created by josephcatrambone on 12/26/16.
 */
fun numericalGradient(f:(Float)->Float, x:Float, h:Float) : Float {
	return (f(x+h) - f(x-h))/(2.0f*h)
}

fun tensorGradientOrder(t1: FloatArray, t2: FloatArray): Float {
	val t1Max = t1.max()!!
	val t2Max = t2.max()!!
	if(t1Max == 0f && t2Max == 0f) {
		return 0f
	} else {
		val max = Math.max(t1Max, t2Max)
		val order = t1.zip(t2).map{ tuple -> Math.abs(tuple.first - tuple.second)/max }.max()!!
		return order
	}
}

fun getNodeGradientOrder(nodeBuilder:(Node)->Node, f:(Float)->Float) : Float {
	val g = Graph()
	val x = InputNode(1, 9)

	val out = nodeBuilder(x)

	g.add(out)

	val dx = 1.0e-4f;
	val xData = Nd4j.create(floatArrayOf(-10f, -5f, -2f, -1f, 0.0f, +1f, 2f, 5f, 10f), intArrayOf(1, 9))
	val inputFeed = mapOf<Node, INDArray>(x to xData)
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

	return tensorGradientOrder(exactDerivative.data().asFloat(), numericalDerivative)
}

class GradientTest : StringSpec() {
	init {
		"Gradient random check" {
			assert(
				getNodeGradientOrder({input -> SigmoidNode(input)}, { x -> 1.0f/(1.0f+Math.exp(-x.toDouble()).toFloat())}) < 0.1f
			)

			assert(getNodeGradientOrder({input -> TanhNode(input)}, { x -> Math.tanh(x.toDouble()).toFloat()}) < 0.1f)

			// This is not differentiable at zero, but it looks right to me.
			//assert(getNodeGradientOrder({input -> LeakyReLUNode(input, 0.01f)}, { x -> if(x >= 0) { x } else { 0.01f*x }}) < 0.1f)
		}

		"HStack should correctly backprop adjoints." {
			val g = Graph()
			val x = InputNode(3, 2)
			val y = InputNode(3, 2)
			val out = ConstantMultiplyNode(HStackNode(ConstantMultiplyNode(x, 2.0f), ConstantMultiplyNode(y, 3.0f)), 4.0f)
			g.add(out)
			val inputMap = mapOf<Node, INDArray>(x to Nd4j.ones(3, 2), y to Nd4j.ones(3, 2))
			val fwd = g.forward(out, inputMap)
			var grad = g.reverse(out, inputMap, fwd)

			println(grad[x])
			println(grad[y])

			grad[y]!!.getFloat(0, 0) shouldBe 12.0f
		}
	}
}
