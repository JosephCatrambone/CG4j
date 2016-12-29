package com.josephcatrambone.cg4j

import io.kotlintest.properties.Gen
import io.kotlintest.specs.StringSpec
import java.util.*

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
			//forAll(Gen.int(), Gen.int(), { a: Int, b: Int ->
			forAll(Gen.choose(0, 100), Gen.choose(0, 100), { a: Int, b: Int ->
				t[a, b] == 1.0f
			})
		}

		"Tensor multiply check." {
			val a = Tensor.ones(3, 5)
			val b = Tensor(shape=intArrayOf(5, 3), data=floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f))
			val c = a.mmul(b)

			println(a)
			println(b)
			println(c)

			c.data[0] shouldBe 35.0f
			c.data[1] shouldBe 40.0f
			c.data[2] shouldBe 45.0f

			c.data[3] shouldBe 35.0f
			c.data[4] shouldBe 40.0f
			c.data[5] shouldBe 45.0f

			c.data[6] shouldBe 35.0f
			c.data[7] shouldBe 40.0f
			c.data[8] shouldBe 45.0f
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

		"Tensor should be able to serialize to and from string." {
			val random = Random()
			val t1 = Tensor.newFromFun(5, 4, 3, 2, initFunction = { i -> random.nextFloat() })
			val s = t1.toString()
			val t2 = Tensor.fromString(s)

			Arrays.deepEquals(t1.shape.toTypedArray(), t2.shape.toTypedArray()) shouldBe true
			Arrays.deepEquals(t1.data.toTypedArray(), t2.data.toTypedArray()) shouldBe true
		}
	}
}