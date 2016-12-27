package com.josephcatrambone.cg4j

import io.kotlintest.properties.Gen
import io.kotlintest.specs.StringSpec

/**
 * Created by josephcatrambone on 12/26/16.
 */
class GraphTest : StringSpec() {
	init {
		"Tensor ones (5, 5, 5) should have 125 elements." {
			Tensor.ones(5, 5, 5).data.size shouldBe 125
		}

		"Serialization Test" {
			val g = Graph()
			val i = InputNode(1, 2, 3)
			val j = InputNode(1, 2, 3)
			val k = AddNode(i, j)
			g.add(k)


		}

	}
}