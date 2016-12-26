package com.josephcatrambone.cg4j

import io.kotlintest.properties.Gen
import io.kotlintest.specs.StringSpec

/**
 * Created by josephcatrambone on 12/26/16.
 */
class TensorTestBasics : StringSpec() {
	init {
		"Tensor ones (5, 5, 5) should have 125 elements." {
			Tensor.ones(5, 5, 5).data.size shouldBe 125
		}

		"Tensor ones (5, 5, 5) should be all one." {
			Tensor.ones(5, 5, 5).data.forEach { x -> x shouldBe 1.0f }
		}

		"Tensor zeros (10, 5, 5, 5, 5) should be all zero." {
			Tensor.zeros(10, 5, 5, 5).data.forEach { x -> x shouldBe 0.0f }
		}

		"Tensor random index check" {
			val t = Tensor.ones(100, 100)
			forAll(Gen.int(), Gen.int(), { a: Int, b: Int ->
				t[a%100, b%100] == 1.0f
			})
		}

		"Tensor stupid activities because the developer is a lazy shit who doesn't write proper tests" {
			var r = Tensor.ones(3, 5)
			var s = r.transpose()

			var t = Tensor.ones(6, 4, 5)
			t[3, 0, 0] = 0.0f
			t[3, 1, 1] = 1.0f
			t[3, 2, 2] = 2.0f
			var u = Tensor.ones(5, 8, 10)
			var v = t.mmul(u)
			println(v.shape)

			println(t.slice((2..4), (0..3), (0..3)))
			t.setSlice((0..3), (0..2), (0..5), value=Tensor.zeros(4, 3, 6))
			println(t)

			var i = Tensor.eye(5, 5)
			var j = Tensor.newFromFun(5, 5, initFunction={ x -> x.toFloat() })
			var k = i.mmul(j)
		}
	}
}