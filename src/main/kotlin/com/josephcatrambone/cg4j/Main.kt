package com.josephcatrambone.cg4j

import java.io.File
import java.util.*

fun main(args : Array<String>) {
	//linearRegression()
	//linearRegression2()
	testRNN()
	//learnXOR()
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

fun testRNN() {
	var x = Tensor(intArrayOf(6, 3), floatArrayOf(
		0f, 0f, 0f,
		0f, 0f, 1f,
		0f, 1f, 0f,
		0f, 1f, 0f,
		1f, 0f, 0f,
		1f, 0f, 0f
	))
	var y = Tensor.zeros(6, 3)
	y.setSlice((0..4), (0..2), value=x.slice((1..5), (0..2)))

	val rnn = RNN(3, 5)
	val startTime = System.currentTimeMillis()
	for(i in (0..100000)) {
		rnn.fit(x, y, 0.1f)
	}
	val endTime = System.currentTimeMillis()
	val dTime = endTime - startTime
	println("Time for n cycles: $dTime")
	val pred = rnn.predict(5)
	for(p in pred.second) {
		println(p.joinToString())
	}
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

	g.save(File("outputXOR.txt"))
}
