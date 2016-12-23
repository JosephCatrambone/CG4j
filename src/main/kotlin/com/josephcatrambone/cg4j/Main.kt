package com.josephcatrambone.cg4j

import java.util.*

fun main(args : Array<String>) {
	//var a: INDArray = Nd4j.create(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f), intArrayOf(2, 3))
	//println("Hello world: $a");

	var r = Tensor.ones(3, 5)
	var s = r.transpose()

	var t = Tensor.ones(6, 4, 5)
	t[3, 0, 0] = 0.0f
	t[3, 1, 1] = 1.0f
	t[3, 2, 2] = 2.0f
	var u = Tensor.ones(5, 8, 10)
	var v = t.mmul(u)
	println(v.shape)

	var i = Tensor.eye(5, 5)
	var j = Tensor.newFromFun(5, 5, initFunction={ x -> x.toFloat() })
	var k = i.mmul(j)

	gradCheck()
	//linearRegression()
	//linearRegression2()
	//testRNN()
	learnXOR()
}

fun gradCheck() {
	fun numericalGradient(f:(Float)->Float, x:Float, h:Float) : Float {
		return (f(x+h) - f(x-h))/(2.0f*h)
	}

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
	val out = g.constantMultiply(x, 10f)
	val f : (Float)->Float = {x -> x*10f}

	val dx = 1.0e-4f;
	val xData = Tensor(shape=intArrayOf(1, 9), data=floatArrayOf(-10f, -5f, -2f, -1f, 0.0f, +1f, 2f, 5f, 10f))
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

	val x = InputNode(1, 1)
	val m = InputNode(1, 1)
	val b = InputNode(1, 1)
	val out = AddNode(ElementMultiplyNode(x, m), b)

	val y = InputNode(1, 1)
	val error = SubtractNode(y, out)
	val squaredError = ElementMultiplyNode(error, error)
	g.add(squaredError)

	val mData = Tensor(intArrayOf(1, 1), floatArrayOf(0.1f))
	val bData = Tensor(intArrayOf(1, 1), floatArrayOf(0.5f))

	val random = Random()
	for(i in (0..1000)) {
		val xData = Tensor(intArrayOf(1, 1), floatArrayOf(random.nextFloat()))
		val yData = Tensor(intArrayOf(1, 1), floatArrayOf((xData[0, 0]*5)-13))
		val inputFeed : Map<Node, Tensor> = mapOf(
			x to xData,
			y to yData,

			m to mData,
			b to bData
		)

		val fwd = g.forward(squaredError, inputFeed)
		val grad = g.reverse(squaredError, inputFeed, fwd)
		mData.sub_i(grad[m]!!.mul(0.1f))
		bData.sub_i(grad[b]!!.mul(0.1f))
		println("Actual: ${yData[0]} -- Predicted: ${fwd[out]}")
	}

	println("Predict y = ${mData[0]}*x + ${bData[0]}")
}

fun linearRegression2() {
	val g = Graph()

	val x = InputNode(1, 2)
	val m = InputNode(2, 1)
	val out = MatrixMultiplyNode(x, m)

	val y = InputNode(1, 1)
	val error = SubtractNode(y, out)
	val squaredError = ElementMultiplyNode(error, error)

	g.add(squaredError)

	val mData = Tensor(intArrayOf(2, 1), floatArrayOf(0.1f, 0.2f))

	val random = Random()
	for(i in (0..1000)) {
		val xData = Tensor(intArrayOf(1, 1), floatArrayOf(random.nextFloat(), 1.0f))
		val yData = Tensor(intArrayOf(1, 1), floatArrayOf((xData[0]*5)-13))
		val inputFeed : Map<Node, Tensor> = mapOf(
				x to xData,
				y to yData,
				m to mData
		)

		val fwd = g.forward(squaredError, inputFeed)
		val grad = g.reverse(squaredError, inputFeed, fwd)
		mData.sub_i(grad[m]!!.mul(0.1f))
		println("Actual: ${yData[0]} -- Predicted: ${fwd[out]}")
	}

	println("$mData")
}

fun learnXOR() {
	val LEARNING_RATE = 0.05f
	val ITERATIONS = 10000
	val BATCH_SIZE = 1

	// Inputs
	val x = InputNode(1, 2)
	val y = InputNode(1, 1)
	// Variables
	val w_xh = VariableNode(2, 3)
	val w_hy = VariableNode(3, 1)
	val b_h = VariableNode(3, 1)
	val b_y = VariableNode(1, 1)
	// Structure
	val hidden_preact = MatrixMultiplyNode(x, w_xh)
	val hidden_biased = AddNode(hidden_preact, b_h) // TODO: Need broadcast-add
	val hidden = TanhNode(hidden_biased)
	val output_preact = MatrixMultiplyNode(hidden, w_hy)
	val output_biased = AddNode(output_preact, b_y)
	val out = TanhNode(output_biased)
	// Error calculation
	val difference = SubtractNode(y, out)
	val squaredError = ElementMultiplyNode(difference, difference)

	// Add everything to the graph.
	val g = Graph()
	g.add(squaredError)

	// Allocate data.
	w_xh.value = Tensor.random(2, 3).mul(0.1f)
	w_hy.value = Tensor.random(3, 1).mul(0.1f)
	b_h.value = Tensor.zeros(3, 1)
	b_y.value = Tensor.zeros(1, 1)

	// Iteratively fix.
	for(iteration in (0..ITERATIONS)) {
		val xData = Tensor.random(BATCH_SIZE, 2)
		val yData = Tensor.zeros(BATCH_SIZE, 1)
		val i = 0
		//for(i in (0..BATCH_SIZE-1)) {
		xData[i, 0] = if(xData[i, 0] > 0.5) { 1.0f } else { 0.0f }
		xData[i, 1] = if(xData[i, 1] > 0.5) { 1.0f } else { 0.0f }
		val label = if ((xData[i, 0] > 0.5) xor (xData[i, 1] > 0.5)) { 1.0f } else { 0.0f }
		yData[i, 0] = label
		//}
		val inputMap = mapOf<Node,Tensor>(
				x to xData,
				y to yData
		)
		val fwd = g.forward(squaredError, inputMap)
		val grad = g.reverse(squaredError, inputMap, fwd)

		w_xh.value.sub_i(grad[w_xh]!!.mul(LEARNING_RATE))
		w_hy.value.sub_i(grad[w_hy]!!.mul(LEARNING_RATE))
		b_h.value.sub_i(grad[b_h]!!.mul(LEARNING_RATE))
		// Currently, including bias makes everything go screwy.  TODO: Problem in bias init or bias add?
		//biasData_y.subi(grad[b_y]!!.mul(LEARNING_RATE))

		if(iteration % 10 == 0) {
			print(g.getOutput(out, mapOf(
					x to Tensor(intArrayOf(1, 2), floatArrayOf(0.0f, 0.0f))
			))[0])
			print("\t")
			print(g.getOutput(out, mapOf(
					x to Tensor(intArrayOf(1, 2), floatArrayOf(0.0f, 1.0f))
			))[0])
			print("\t")
			print(g.getOutput(out, mapOf(
					x to Tensor(intArrayOf(1, 2), floatArrayOf(1.0f, 0.0f))
			))[0])
			print("\t")
			print(g.getOutput(out, mapOf(
					x to Tensor(intArrayOf(1, 2), floatArrayOf(1.0f, 1.0f))
			))[0])
			println()
		}
	}
}
