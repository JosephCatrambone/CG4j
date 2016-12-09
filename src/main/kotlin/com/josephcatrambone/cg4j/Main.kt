package com.josephcatrambone.cg4j

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.util.*

fun main(args : Array<String>) {
	//var a: INDArray = Nd4j.create(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f), intArrayOf(2, 3))
	//println("Hello world: $a");

	gradCheck()
	//linearRegression()
	//linearRegression2()
	//learnXOR()
}

fun gradCheck() {
	fun numericalGradient(f:(Float)->Float, x:Float, h:Float) : Float {
		return (f(x+h) - f(x-h))/(2.0f*h)
	}

	val g = Graph()
	val x = g.addInput(1, 9)

	// Test tanh.
	//val out = g.addTanh(x)
	//val f : (Float)->Float = { x -> Math.tanh(x.toDouble()).toFloat() };
	// Test x^2
	//val out = g.addElementMultiply(x, x)
	//val f : (Float)->Float = { x -> x*x };
	// Test x+x
	//val out = g.addAdd(x, x)
	//val f : (Float)->Float = { x -> x+x };
	// Test x-x
	//val out = g.addSubtract(x, x)
	//val f : (Float)->Float = {x -> x-x } // Basically zero.  d/dx x wrt x = 1.  d/dx -x wrt x = -1.  1 + -1 = 0.
	// Test 1/x
	//val out = g.addPower(x, -1f)
	//val f : (Float)->Float = {x -> Math.pow(x.toDouble(), -1.0).toFloat() }
	// Test something else.
	val out = g.addMultiplyConstant(x, 10f)
	val f : (Float)->Float = {x -> x*10f}

	val dx = 1.0e-4f;
	val xData = Nd4j.create(floatArrayOf(-10f, -5f, -2f, -1f, 0.0f, +1f, 2f, 5f, 10f))
	val inputFeed = mapOf(x to xData)
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

fun linearRegression() {
	val g = Graph()

	val x = g.addInput(1, 1)
	val m = g.addInput(1, 1)
	val b = g.addInput(1, 1)
	val out = g.addAdd(g.addElementMultiply(x, m), b)

	val y = g.addInput(1, 1)
	val error = g.addSubtract(y, out)
	val squaredError = g.addElementMultiply(error, error)

	val mData = Nd4j.create(floatArrayOf(0.1f))
	val bData = Nd4j.create(floatArrayOf(0.5f))

	val random = Random()
	for(i in (0..1000)) {
		val xData = Nd4j.create(floatArrayOf(random.nextFloat()))
		val yData = Nd4j.create(floatArrayOf((xData.getFloat(0)*5)-13))
		val inputFeed : Map<Int, INDArray> = mapOf(
			x to xData,
			y to yData,

			m to mData,
			b to bData
		)

		val fwd = g.forward(squaredError, inputFeed)
		val grad = g.reverse(squaredError, inputFeed, fwd)
		mData.subi(grad[m]!!.mul(0.1f))
		bData.subi(grad[b]!!.mul(0.1f))
		println("Actual: ${yData.getFloat(0)} -- Predicted: ${fwd[out]}")
	}

	println("Predict y = ${mData.getFloat(0)}*x + ${bData.getFloat(0)}")
}

fun linearRegression2() {
	val g = Graph()

	val x = g.addInput(1, 2)
	val m = g.addInput(2, 1)
	val out = g.addMatrixMultiply(x, m)

	val y = g.addInput(1, 1)
	val error = g.addSubtract(y, out)
	val squaredError = g.addElementMultiply(error, error)

	val mData = Nd4j.create(arrayOf(floatArrayOf(0.1f), floatArrayOf(0.2f)))

	val random = Random()
	for(i in (0..1000)) {
		val xData = Nd4j.create(floatArrayOf(random.nextFloat(), 1.0f))
		val yData = Nd4j.create(floatArrayOf((xData.getFloat(0)*5)-13))
		val inputFeed : Map<Int, INDArray> = mapOf(
				x to xData,
				y to yData,
				m to mData
		)

		val fwd = g.forward(squaredError, inputFeed)
		val grad = g.reverse(squaredError, inputFeed, fwd)
		mData.subi(grad[m]!!.mul(0.1f))
		println("Actual: ${yData.getFloat(0)} -- Predicted: ${fwd[out]}")
	}

	println("$mData")
}

fun learnXOR() {
	val LEARNING_RATE = 0.01f
	val ITERATIONS = 100000
	val BATCH_SIZE = 1

	val g = Graph()
	// Inputs
	val x = g.addInput(1, 2)
	val y = g.addInput(1, 1)
	// Variables
	val w_xh = g.addInput(2, 3)
	val w_hy = g.addInput(3, 1)
	val b_h = g.addInput(3, 1)
	val b_y = g.addInput(1, 1)
	// Structure
	val hidden_preact = g.addMatrixMultiply(x, w_xh)
	val hidden_biased = g.addAdd(hidden_preact, b_h) // TODO: Need broadcast-add
	val hidden = g.addTanh(hidden_biased)
	val output_preact = g.addMatrixMultiply(hidden, w_hy)
	val output_biased = g.addAdd(output_preact, b_y)
	val out = g.addTanh(output_biased)
	// Error calculation
	val difference = g.addSubtract(y, out)
	val squaredError = g.addElementMultiply(difference, difference)

	// Allocate data.
	val weightData_xh = Nd4j.rand(2, 3).mul(0.1)
	val weightData_hy = Nd4j.rand(3, 1).mul(0.1)
	val biasData_h = Nd4j.zeros(3, 1)
	val biasData_y = Nd4j.zeros(1, 1)

	// Iteratively fix.
	for(iteration in (0..ITERATIONS)) {
		val xData = Nd4j.rand(BATCH_SIZE, 2)
		val yData = Nd4j.zeros(BATCH_SIZE, 1)
		val i = 0
		//for(i in (0..BATCH_SIZE-1)) {
			xData.put(i, 0, if(xData.getDouble(i, 0) > 0.5) { 1.0 } else { 0.0 })
			xData.put(i, 1, if(xData.getDouble(i, 1) > 0.5) { 1.0 } else { 0.0 })
			val label = if ((xData.getDouble(i, 0) > 0.5) xor (xData.getDouble(i, 1) > 0.5)) { 1.0f } else { 0.0f }
			yData.put(i, 0, label)
		//}
		val inputMap = mapOf(
			x to xData,
			y to yData,
			w_xh to weightData_xh,
			w_hy to weightData_hy,
			b_h to biasData_h,
			b_y to biasData_y
		)
		val fwd = g.forward(squaredError, inputMap)
		val grad = g.reverse(squaredError, inputMap, fwd)

		weightData_xh.subi(grad[w_xh]!!.mul(LEARNING_RATE))
		weightData_hy.subi(grad[w_hy]!!.mul(LEARNING_RATE))
		biasData_h.subi(grad[b_h]!!.mul(LEARNING_RATE))
		// Currently, including bias makes everything go screwy.  TODO: Problem in bias init or bias add?
		//biasData_y.subi(grad[b_y]!!.mul(LEARNING_RATE))

		print(g.getOutput(out, mapOf(
				x to Nd4j.create(floatArrayOf(0.0f, 0.0f)),
				w_xh to weightData_xh,
				w_hy to weightData_hy,
				b_h to biasData_h,
				b_y to biasData_y
		)))
		print("\t")
		print(g.getOutput(out, mapOf(
				x to Nd4j.create(floatArrayOf(1.0f, 0.0f)),
				w_xh to weightData_xh,
				w_hy to weightData_hy,
				b_h to biasData_h,
				b_y to biasData_y
		)))
		print("\t")
		print(g.getOutput(out, mapOf(
				x to Nd4j.create(floatArrayOf(1.0f, 0.0f)),
				w_xh to weightData_xh,
				w_hy to weightData_hy,
				b_h to biasData_h,
				b_y to biasData_y
		)))
		print("\t")
		print(g.getOutput(out, mapOf(
				x to Nd4j.create(floatArrayOf(1.0f, 1.0f)),
				w_xh to weightData_xh,
				w_hy to weightData_hy,
				b_h to biasData_h,
				b_y to biasData_y
		)))
		println()

	}

}
