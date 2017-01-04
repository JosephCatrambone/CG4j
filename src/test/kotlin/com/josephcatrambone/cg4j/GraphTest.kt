package com.josephcatrambone.cg4j

import io.kotlintest.properties.Gen
import io.kotlintest.specs.StringSpec
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
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
			val LEARNING_RATE = 0.1f
			val ITERATIONS = 20000
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
			w_xh.value = Nd4j.rand(2, 3).mul(0.1f)
			w_hy.value = Nd4j.rand(3, 1).mul(0.1f)
			b_h.value = Nd4j.zeros(3, 1)
			b_y.value = Nd4j.zeros(1, 1)

			// Iteratively fix.
			for(iteration in (0..ITERATIONS)) {
				val xData = Nd4j.rand(BATCH_SIZE, 2)
				val yData = Nd4j.zeros(BATCH_SIZE, 1)
				for(i in (0..BATCH_SIZE-1)) {
					xData.put(i, 0, if (xData.getFloat(i, 0) > 0.5f) { 1.0f } else { 0.0f })
					xData.put(i, 1, if (xData.getFloat(i, 1) > 0.5f) { 1.0f } else { 0.0f })
					val label = if ((xData.getFloat(i, 0) > 0.5f) xor (xData.getFloat(i, 1) > 0.5)) { 1.0f } else { 0.0f }
					yData.put(i, 0, label)
				}
				//}
				val inputMap = mapOf<Node, INDArray>(
						x to xData,
						y to yData
				)
				val fwd = g.forward(squaredError, inputMap)
				val grad = g.reverse(squaredError, inputMap, fwd)

				w_xh.value.subi(grad[w_xh]!!.mul(LEARNING_RATE))
				w_hy.value.subi(grad[w_hy]!!.mul(LEARNING_RATE))
				b_h.value.subi(grad[b_h]!!.mul(LEARNING_RATE))
				b_y.value.subi(grad[b_y]!!.mul(LEARNING_RATE))
			}

			assert(g.getOutput(out, mapOf(x to Nd4j.create(floatArrayOf(0f, 0f), intArrayOf(1, 2)))).getFloat(0) < 0.1f)
			assert(g.getOutput(out, mapOf(x to Nd4j.create(floatArrayOf(0f, 1f), intArrayOf(1, 2)))).getFloat(0) > 0.9f)
			assert(g.getOutput(out, mapOf(x to Nd4j.create(floatArrayOf(1f, 0f), intArrayOf(1, 2)))).getFloat(0) > 0.9f)
			assert(g.getOutput(out, mapOf(x to Nd4j.create(floatArrayOf(1f, 1f), intArrayOf(1, 2)))).getFloat(0) < 0.1f)

			//g.save(File("outputXOR.txt"))
		}
	}
}