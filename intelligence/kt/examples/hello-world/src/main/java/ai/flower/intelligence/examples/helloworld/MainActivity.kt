package ai.flower.intelligence.example.helloworld

import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.LinearLayoutCompat

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val layout = LinearLayoutCompat(this).apply {
            orientation = LinearLayoutCompat.VERTICAL
            gravity = Gravity.CENTER
            setBackgroundColor(Color.WHITE)
        }

        val textView = TextView(this).apply {
            text = "Hello, World!"
            textSize = 24f
            setTextColor(Color.BLACK)
        }

        layout.addView(textView)
        setContentView(layout)
    }
}
