package com.josephcatrambone.cg4j

import io.kotlintest.properties.Gen
import io.kotlintest.specs.StringSpec
import java.io.File

/**
 * Created by josephcatrambone on 12/26/16.
 */
class GraphTest : StringSpec() {
	init {
		"Serialization Test" {
			val g = Graph()
			val i = InputNode(1, 2, 3)
			val j = InputNode(1, 2, 3)
			val k = AddNode(i, j)
			g.add(k)
		}

		"Verify XOR" {
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
			//val squaredError = ElementMultiplyNode(difference, difference)
			val squaredError = PowerNode(difference, 2.0f)

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
				b_y.value.sub_i(grad[b_y]!!.mul(LEARNING_RATE))
			}

			assert(g.getOutput(out, mapOf(x to Tensor(intArrayOf(1, 2), floatArrayOf(0f, 0f)))).data[0] < 0.1f)
			assert(g.getOutput(out, mapOf(x to Tensor(intArrayOf(1, 2), floatArrayOf(1f, 0f)))).data[0] > 0.9f)
			assert(g.getOutput(out, mapOf(x to Tensor(intArrayOf(1, 2), floatArrayOf(0f, 1f)))).data[0] > 0.9f)
			assert(g.getOutput(out, mapOf(x to Tensor(intArrayOf(1, 2), floatArrayOf(1f, 1f)))).data[0] < 0.1f)

			//g.save(File("outputXOR.txt"))
		}
	}
}