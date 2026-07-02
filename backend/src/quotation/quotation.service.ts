import { Injectable, InternalServerErrorException, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';
import * as fs from 'fs';
import { Quotation } from './schemas/quotation.schema';

const execAsync = promisify(exec);

@Injectable()
export class QuotationService {
  constructor(
    @InjectModel(Quotation.name) private quotationModel: Model<Quotation>,
  ) {}

  async processPdfAndCompare(file: any): Promise<any> {
    try {
      const workspaceDir = path.resolve(__dirname, '../../../');
      const uploadsDir = path.join(workspaceDir, 'uploads');

      // Create uploads directory if not exists
      if (!fs.existsSync(uploadsDir)) {
        fs.mkdirSync(uploadsDir, { recursive: true });
      }

      const filePath = path.join(uploadsDir, file.originalname || 'uploaded_quotation.pdf');
      
      // Write the uploaded buffer payload to temp file
      fs.writeFileSync(filePath, file.buffer);
      
      const absoluteFilePath = path.resolve(filePath);
      
      // Run Python AI Extraction Pipeline
      // We pass the file path as an argument to infer_and_compare.py
      const command = `python3 infer_and_compare.py "${absoluteFilePath}"`;
      const { stdout, stderr } = await execAsync(command, { cwd: workspaceDir });

      if (stderr) {
        console.warn('Python Pipeline Warning/Stderr:', stderr);
      }

      // Try to parse stdout to JSON.
      let extractedData: any;
      try {
        extractedData = JSON.parse(stdout);
      } catch {
        // Fallback: search for json pattern in stdout
        const match = stdout.match(/\{[\s\S]*\}/);
        if (match) {
          extractedData = JSON.parse(match[0]);
        } else {
          throw new Error(`Could not parse JSON from output: ${stdout}`);
        }
      }

      // Save to MongoDB for Continuous Learning and Audit
      const newQuotation = new this.quotationModel({
        originalFile: absoluteFilePath,
        extractedData: extractedData,
        status: 'PENDING_REVIEW',
      });
      await newQuotation.save();

      return newQuotation;

    } catch (error) {
      throw new InternalServerErrorException(`AI Pipeline Failed: ${error.message}`);
    }
  }

  async getAllQuotations(): Promise<Quotation[]> {
    return this.quotationModel.find().sort({ createdAt: -1 }).exec();
  }

  async correctAndRetrain(id: string, correctedData: any): Promise<any> {
    const quotation = await this.quotationModel.findByIdAndUpdate(
      id,
      { extractedData: correctedData, status: 'VERIFIED' },
      { new: true }
    );
    
    if (!quotation) {
      throw new NotFoundException(`Quotation with ID ${id} not found`);
    }

    const workspaceDir = path.resolve(__dirname, '../../../');
    
    // Write a script or trigger dataset update (e.g. append correction to training dataset)
    // We can write a script or append to train_data.json directly.
    const trainingFile = path.join(workspaceDir, 'train_data.json');
    try {
      if (fs.existsSync(trainingFile)) {
        const fileContent = fs.readFileSync(trainingFile, 'utf-8');
        const trainData = JSON.parse(fileContent);
        
        // Prepare tokens and bboxes from corrected data (dummy alignment or simplified layout generation)
        // In a real application, we would align tokens and corrected attributes.
        // For demonstration, we append the new labeled instance to dataset.
        if (correctedData.tokens && correctedData.ner_tags && correctedData.bboxes) {
          trainData.push({
            tokens: correctedData.tokens,
            ner_tags: correctedData.ner_tags,
            bboxes: correctedData.bboxes
          });
          fs.writeFileSync(trainingFile, JSON.stringify(trainData, null, 2));
        }
      }
    } catch (e) {
      console.error('Failed to append correction to train_data.json:', e);
    }
    
    // Optionally trigger asynchronous background retraining
    // execAsync('python3 train_layoutlm.py', { cwd: workspaceDir }).catch(console.error);

    return { success: true, message: 'Added to Training Dataset', quotation };
  }
}
