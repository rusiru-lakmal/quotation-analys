import { Controller, Post, Body, Get, Put, Param, UploadedFile, UseInterceptors } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { QuotationService } from './quotation.service';

@Controller('api/v1/quotations')
export class QuotationController {
  constructor(private readonly quotationService: QuotationService) {}

  // 1. Upload PDF and process via Python Engine
  @Post('process')
  @UseInterceptors(FileInterceptor('file'))
  async processQuotation(@UploadedFile() file: any) {
    return this.quotationService.processPdfAndCompare(file);
  }

  @Get()
  async getQuotations() {
    return this.quotationService.getAllQuotations();
  }

  // 2. Submit corrections and retrain LayoutLM
  @Put(':id/correct')
  async submitCorrection(
    @Param('id') id: string,
    @Body() correctedData: any,
  ) {
    return this.quotationService.correctAndRetrain(id, correctedData);
  }
}
