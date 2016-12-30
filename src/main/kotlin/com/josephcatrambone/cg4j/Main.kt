package com.josephcatrambone.cg4j

import java.io.File
import java.util.*

fun main(args : Array<String>) {
	//linearRegression()
	//linearRegression2()
	testRNN()
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
	var x = Tensor(intArrayOf(6, 5), floatArrayOf(
		1f, 0f, 0f, 0f, 0f,
		0f, 1f, 0f, 0f, 0f,
		0f, 0f, 1f, 0f, 0f,
		0f, 0f, 1f, 0f, 0f,
		0f, 0f, 0f, 1f, 0f,
		0f, 0f, 0f, 0f, 1f
			))
	var y = Tensor.zeros(6, 5)
	y.setSlice((0..4), (0..4), value=x.slice((1..5), (0..4)))

	val rnn = RNN(5, 20)
	val startTime = System.currentTimeMillis()
	var learningRate = 0.2f
	for(i in (0..50000)) {
		rnn.fit(x, y, learningRate)
		//learningRate *= 0.9999f
	}
	println(learningRate)
	val endTime = System.currentTimeMillis()
	val dTime = endTime - startTime
	println("Time for n cycles: $dTime")
	val pred = rnn.predict(10)
	for(p in pred.second) {
		println(p.joinToString("\t"))
	}
}
