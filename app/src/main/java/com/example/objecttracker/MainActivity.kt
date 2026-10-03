package com.example.objecttracker
import android.Manifest
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot

class MainActivity : AppCompatActivity() {
    lateinit var previewView: PreviewView
    lateinit var tvInfo: TextView
    var targetHsv = Scalar(0.0,0.0,0.0)
    var isTracking = false
    var lastCenter: Point? = null
    var lastTime = System.currentTimeMillis()
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        tvInfo = findViewById(R.id.tvInfo)
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "فشل OpenCV", Toast.LENGTH_LONG).show()
        }
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 101)
        previewView.setOnTouchListener { _, e ->
            if (e.action == android.view.MotionEvent.ACTION_DOWN) {
                isTracking =!isTracking
                lastCenter = null
                lastTime = System.currentTimeMillis()
                tvInfo.text = if(isTracking) "يتتبع..." else "متوقف - المس للبدء"
            }
            true
        }
        startCamera()
    }
    fun startCamera(){
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also{ it.setSurfaceProvider(previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).setTargetResolution(android.util.Size(640,480)).build()
            analysis.setAnalyzer(ContextCompat.getMainExecutor(this)){ proxy ->
                process(proxy)
                proxy.close()
            }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }
    fun process(proxy: ImageProxy){
        if(!isTracking) return
        val rgbMat = yuvToRgbMat(proxy)?: return
        val hsv = Mat()
        Imgproc.cvtColor(rgbMat, hsv, Imgproc.COLOR_RGB2HSV)
        if(lastCenter==null){
            val cc = hsv.get(hsv.rows()/2, hsv.cols()/2)
            if(cc!=null) targetHsv = Scalar(cc[0],cc[1],cc[2])
        }
        val mask = Mat()
        Core.inRange(hsv, Scalar(targetHsv.`val`[0]-12,60.0,60.0), Scalar(targetHsv.`val`[0]+12,255.0,255.0), mask)
        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(mask, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        if(contours.isNotEmpty()){
            val biggest = contours.maxByOrNull{ Imgproc.contourArea(it)}!!
            val area = Imgproc.contourArea(biggest)
            if(area>800){
                val rect = Imgproc.boundingRect(biggest)
                val cx = rect.x+rect.width/2; val cy = rect.y+rect.height/2
                val curr = Point(cx.toDouble(), cy.toDouble())
                val delta = lastCenter?.let{ hypot(curr.x-it.x, curr.y-it.y)}?:0.0
                val dt = (System.currentTimeMillis()-lastTime)/1000.0
                val vel = if(dt>0) delta/dt else 0.0
                runOnUiThread{ tvInfo.text = "X:$cx Y:$cy\nArea:${area.toInt()} Delta:${delta.toInt()} Vel:${vel.toInt()}"}
                lastCenter=curr; lastTime=System.currentTimeMillis()
            }
        }
        rgbMat.release(); hsv.release(); mask.release()
    }
    fun yuvToRgbMat(proxy: ImageProxy): Mat? {
        try{
            val yB = proxy.planes[0].buffer; val uB = proxy.planes[1].buffer; val vB = proxy.planes[2].buffer
            val yS = yB.remaining(); val uS = uB.remaining(); val vS = vB.remaining()
            val nv21 = ByteArray(yS+uS+vS)
            yB.get(nv21,0,yS); vB.get(nv21,yS,vS); uB.get(nv21,yS+vS,uS)
            val yuv = Mat(proxy.height+proxy.height/2, proxy.width, CvType.CV_8UC1)
            yuv.put(0,0,nv21); val rgb = Mat()
            Imgproc.cvtColor(yuv, rgb, Imgproc.COLOR_YUV2RGB_NV21)
            yuv.release(); return rgb
        }catch(e: Exception){ return null }
    }
}
